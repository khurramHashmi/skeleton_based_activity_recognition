# Third Party
import torch
import torch.nn as nn


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.num_features))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2 * input_dim, output_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # self.perceptrons = nn.ModuleList()
        # for _ in range(seq_len):
        #     self.perceptrons.append(nn.Linear(self.hidden_dim, output_dim))

        self.dense_layers = torch.rand(
            (self.hidden_dim, output_dim),
            dtype=torch.float,
            requires_grad=True
        )

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        # output_seq = torch.empty(
        #     self.seq_len,
        #     self.output_dim,
        #     dtype=torch.float
        # )
        # for index, perceptron in zip(range(self.seq_len), self.perceptrons):
        #     output_seq[index] = perceptron(x[index])
        #
        # return output_seq

        return torch.mm(x, self.dense_layers)


#########
# EXPORTS
#########


class RAE(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(RAE, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, num_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class simple_autoencoder(nn.Module):
    def __init__(self):
        super(simple_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 78),
            nn.ReLU(True),
            nn.Linear(78, 64),
            nn.ReLU(True),
            nn.Linear(64, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 78),
            nn.ReLU(True),
            nn.Linear(78, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


'''
AutoEncoder With Classification Network with Video  For 128 Embedding Size for Previous Training3
CODE starts here
'''
class SimpleAutoEncoderVideo_128(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoderVideo_128, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7500, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128))
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 7500),
            nn.ReLU(True))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class classification_network_128(nn.Module):

    def __init__(self, num_feature, num_class):
        super(classification_network_128, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)


    def forward(self, x):

        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x
'''
AutoEncoder With Classification Network with Video For 128 Embedding Size for Previous Training
ENDS HERE
'''


class SimpleAutoEncoderVideo(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoderVideo, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7500, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512))
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 7500),
            nn.ReLU(True))


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


'''
    classification network working in the supervised manner
    backpropogating the loss
'''


class classification_network(nn.Module):

    def __init__(self, num_feature, num_class):
        super(classification_network, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 1024)
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, 128)
        self.layer_5 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.batchnorm5 = nn.BatchNorm1d(64)


    def forward(self, x):

        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


'''
Class for Autoencoder with Classification
'''
class AutoEncoderWithClassifier(nn.Module):
    def __init__(self):
        super(AutoEncoderWithClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7500, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 7500),
            nn.ReLU(True))

        self.classifier = classification_network(256, 60)

    def forward(self, x):

        encoded_output = self.encoder(x)
        decoded_output = self.decoder(encoded_output)
        class_output = self.classifier(encoded_output)

        return decoded_output, class_output



'''
Class for Classification network with conv1d

'''

class ClassificationConv1D(nn.Module):

    def __init__(self, num_feature, num_class):
        super(ClassificationConv1D, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 1024)

        self.layer_2 = nn.Conv1d(1024, 512, kernel_size=3)
        self.layer_3 = nn.Conv1d(512, 256, kernel_size=3)
        self.layer_4 = nn.Conv1d(256, 128, kernel_size=3)
        self.layer_5 = nn.Conv1d(128, 64, kernel_size=3)

        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.batchnorm5 = nn.BatchNorm1d(64)


    def forward(self, x):

        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = x.unsqueeze(-1)
        x = x.expand(10, 1024, 7500)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x
