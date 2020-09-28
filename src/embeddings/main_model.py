import torch
import torch.nn as nn

'''
    New classes for the combined model
    AutoEncoder with Skeleton
    AutoEncoder with Videos
    Classification Network At the end
'''

class SkeletonAutoEnoder(nn.Module):

    def __init__(self):
        super(SkeletonAutoEnoder, self).__init__()

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


class VideoAutoEnoder(nn.Module):

    def __init__(self, batch_size):
        super(VideoAutoEnoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 75, 1024),  #Input Dimension depends upon the last layer of Skeleton_Encoder
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
            nn.Linear(1024, batch_size * 75),
            nn.ReLU(True))


class classification_network_128(nn.Module):

    def __init__(self, num_feature, num_class, batch_size):
        super(classification_network_128, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 96)
        self.layer_2 = nn.Linear(96, 80)
        self.layer_3 = nn.Linear(80, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(96)
        self.batchnorm2 = nn.BatchNorm1d(80)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.batch_size = batch_size

        '''
        Defining Skel and Video Auto Encoders
        '''
        self.skel = SkeletonAutoEnoder()
        self.video = VideoAutoEnoder(batch_size)

    def forward(self, x):
        #taking input as a video but then pass it as a skeleton

        x = self.transform_input_for_skeleton(x)

        # Forward pass For Skel_AutoEncoder
        skel_encoded = self.skel.encoder(x)
        skel_decoded = self.skel.decoder(skel_encoded)

        # Forward pass For Video_AutoEncoder
        # Reshaping Skeleten_ecoded embeddings Accordingly
        input_video_encoder = self.transform_input_for_video(skel_encoded)
        video_encoded = self.video.encoder(input_video_encoder)
        video_decoded = self.video.decoder(video_encoded)

        # Reshaping Video Decoded such that skeleton can be reproduced
        video_decoded = self.transform_input_for_skeleton(video_decoded)

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

        class_out = self.layer_out(x)

        return skel_decoded, video_decoded, class_out

    def transform_input_for_skeleton(self, x):

        return x.view(-1,100) # Since the size of skeleton will always be 100

    def transform_input_for_video(self, x):

        return x.view(100, -1) # Since the size of skeleton will always be 100


'''
NOW NEW EXPERIMENT WITH RNN IN AUTO ENCODER ALONG WITH THE SAME CLASSIFIER
'''
############
# COMPONENTS
############





class RecurrentAutoencoderWithClassifier(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64, num_class=60, batch_size=100):
    super(RecurrentAutoencoderWithClassifier, self).__init__()


    self.classifer = classification_network_128(num_feature=n_features, num_class=num_class, batch_size=batch_size)

  def forward(self, x):
    skel_encoded = self.encoder(x)
    x = self.decoder(x)
    return x


class VideoEncoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(VideoEncoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
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

  # def forward(self, x):
  #   x = x.reshape((1, self.seq_len, self.n_features))
  #   x, (_, _) = self.rnn1(x)
  #   x, (hidden_n, _) = self.rnn2(x)
  #   return hidden_n.reshape((self.n_features, self.embedding_dim))


class VideoDecoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(VideoDecoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
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

    # self.output_layer = nn.Linear(self.hidden_dim, n_features)

  # def forward(self, x):
  #   x = x.repeat(self.seq_len, self.n_features)
  #   x = x.reshape((self.n_features, self.seq_len, self.input_dim))
  #   x, (hidden_n, cell_n) = self.rnn1(x)
  #   x, (hidden_n, cell_n) = self.rnn2(x)
  #   x = x.reshape((self.seq_len, self.hidden_dim))
  #   return self.output_layer(x)





class RecurentAutoEncoderWithClassifier(nn.Module):

    def __init__(self, num_feature, num_class, batch_size):
        super(RecurentAutoEncoderWithClassifier, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 96)
        self.layer_2 = nn.Linear(96, 80)
        self.layer_3 = nn.Linear(80, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(96)
        self.batchnorm2 = nn.BatchNorm1d(80)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.batch_size = batch_size

        '''
        Defining Skel and Video Auto Encoders
        '''
        self.skel = SkeletonAutoEnoder()
        self.video_encoder = VideoEncoder(seq_len=32 * 75, input_dim=64, n_features=1)

        self.video_decoder = VideoDecoder(seq_len=32 * 75, input_dim=64, n_features=1)

    def forward(self, x):
        #taking input as a video but then pass it as a skeleton

        x = self.transform_input_for_skeleton(x)

        # Forward pass For Skel_AutoEncoder
        skel_encoded = self.skel.encoder(x)
        skel_decoded = self.skel.decoder(skel_encoded)

        # Forward pass For Video_AutoEncoder
        # Reshaping Skeleten_ecoded embeddings Accordingly
        input_video_encoder = self.transform_input_for_video(skel_encoded)
        video_encoded = self.video.encoder(input_video_encoder)
        video_decoded = self.video.decoder(video_encoded)

        # Reshaping Video Decoded such that skeleton can be reproduced
        video_decoded = self.transform_input_for_skeleton(video_decoded)

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

        class_out = self.layer_out(x)

        return skel_decoded, video_decoded, class_out

    def transform_input_for_skeleton(self, x):

        return x.view(-1,100) # Since the size of skeleton will always be 100

    def transform_input_for_video(self, x):

        return x.view(100, -1) # Since the size of skeleton will always be 100