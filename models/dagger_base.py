import torch
import torch.nn as nn
import torch.nn.functional as F


class DaggerBaseModel(torch.nn.Module):
    def __init__(
            self,
            action_sz=6,
            target_sz=300,
            dropout_rate=0.25,
    ):
        resnet_embedding_sz = 512
        hidden_state_sz = 512
        super(DaggerBaseModel, self).__init__()
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_sz, 64)
        self.embed_action = nn.Linear(action_sz, 10)

        pointwise_in_channels = 138

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_sz
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)
        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_sz)

        # self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        # self.actor_linear.weight.data = norm_col_init(
        #     self.actor_linear.weight.data, 0.01
        # )
        self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = norm_col_init(
        #     self.critic_linear.weight.data, 1.0
        # )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_sz)

        self.dropout = nn.Dropout(p=dropout_rate)

    def embedding(self, state, target, action_probs, params=None):

        action_embedding_input = action_probs

        if params is None:
            glove_embedding = F.relu(self.embed_glove(target))
            glove_reshaped = glove_embedding.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(self.conv1(state))
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = F.relu(self.pointwise(x))
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:
            glove_embedding = F.relu(
                F.linear(
                    target,
                    weight=params["embed_glove.weight"],
                    bias=params["embed_glove.bias"],
                )
            )

            glove_reshaped = glove_embedding.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(
                F.linear(
                    action_embedding_input,
                    weight=params["embed_action.weight"],
                    bias=params["embed_action.bias"],
                )
            )
            action_reshaped = action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(
                F.conv2d(
                    state, weight=params["conv1.weight"], bias=params["conv1.bias"]
                )
            )
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)

            x = F.relu(
                F.conv2d(
                    x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
                )
            )
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden, params=None):
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:
            # hx, cx = self._backend.LSTMCell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            # Change for pytorch 1.01
            hx, cx = nn._VF.lstm_cell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, params=None):

        state = model_input['res18fm']
        (hx, cx) = model_input['hidden']

        target = model_input['glove']

        action_probs = model_input['action_probs']

        x, _ = self.embedding(state, target, action_probs, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

        return dict(
            policy=actor_out,
            value=critic_out,
            hidden=(hx, cx),
        )
        # ModelOutput(
        #     value=critic_out,
        #     logit=actor_out,
        #     hidden=(hx, cx),
        #     embedding=image_embedding,
        # )


if __name__ == "__main__":
    model = DaggerBaseModel(3)
    input_ = {
        'res18fm': torch.randn(4, 512, 7, 7),
        'action_probs': torch.randn(4, 3),
        'hidden': (torch.randn(4, 512), torch.randn(4, 512)),
        'glove': torch.randn(4, 300)
    }

    cc = {}
    for name, param in model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        cc[name] = param_copied
    out = model.forward(input_)
    print(out['value'])
    # out['value'].mean().backward()
    # out = model.forward(input_)
    # print(out['value'])
    out = model.forward(input_)
    print(out['value'])