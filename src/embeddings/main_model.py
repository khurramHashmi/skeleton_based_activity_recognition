import torch.nn.functional as F
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
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 78),
            nn.ReLU(),
            nn.Linear(78, 64),
            nn.ReLU(),
            nn.Linear(64, 32))

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 78),
            nn.ReLU(),
            nn.Linear(78, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU())

        '''Experimenting with 1D CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(100, 78, 3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(78,64,3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(64, 32, 3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32,64,3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxUnpool1d(3),
            nn.ConvTranspose1d(64, 78, 3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxUnpool1d(3),
            nn.ConvTranspose1d(78, 100, 3),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxUnpool1d(3))'''


class VideoAutoEnoder(nn.Module):

    def __init__(self, batch_size):
        super(VideoAutoEnoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 75, 1024),  #Input Dimension depends upon the last layer of Skeleton_Encoder
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128))

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, batch_size * 75),
            nn.ReLU())

        '''Experimenting with 1D CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(32 * 75, 1024, 3),
            nn.MaxPool1d(3),
            nn.Conv1d(1024 ,512 ,3),
            nn.MaxPool1d(3),
            nn.Conv1d(512 , 256, 3),
            nn.MaxPool1d(3),
            nn.Conv1d(256 , 128, 3),
            nn.MaxPool1d(3))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 256 ,3),
            nn.MaxUnpool1d(3),
            nn.ConvTranspose1d(256, 512, 3),
            nn.MaxUnpool1d(3),
            nn.ConvTranspose1d(512, 1024, 3),
            nn.MaxUnpool1d(3),
            nn.ConvTranspose1d(1024, batch_size * 75, 3),
            nn.MaxUnpool1d(3))
            '''


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
       # print(input_video_encoder.shape)
        video_encoded = self.video.encoder(input_video_encoder)
        video_decoded = self.video.decoder(video_encoded)
        print(video_decoded.shape)
        # Reshaping Video Decoded such that skeleton can be reproduced
        video_decoded = video_decoded.view(-1,100) #self.transform_input_for_skeleton(video_decoded)
        print(video_decoded.shape)

        x = self.layer_1(video_encoded)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        class_out = self.layer_out(x)
        # class_out = F.softmax(class_out, dim=-1)

        return skel_decoded, video_decoded, class_out

    def transform_input_for_skeleton(self, x):

        return x.view(-1,150) # Since the size of skeleton will always be 100


    def transform_input_for_video(self, x):
        #print(x.shape)
        return x.view(100, -1) # Since the size of skeleton will always be 100


'''
ANOTHER MODEL ONLY WITH AUTE-ENCODERS UNSUPERVISED
FOR SKELETONS AND VIDEOS WITH MULTIPLE LOSSES
'''
class UnsuperVisedAE(nn.Module):

    def __init__(self, batch_size):
        super(UnsuperVisedAE, self).__init__()

        self.batch_size = batch_size

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

        return skel_decoded, video_decoded

    def transform_input_for_skeleton(self, x):

        return x.view(-1,100) # Since the size of skeleton will always be 100


    def transform_input_for_video(self, x):

        return x.view(100, -1) # Since the size of skeleton will always be 100




















'''
NOW NEW EXPERIMENT WITH RNN IN AUTO ENCODER ALONG WITH THE SAME CLASSIFIER
ONE WITH VIDEO ENCODER/DECODER AS LSTM
ONE WITH VIDEO ENCODER/DECODER AS GRU
'''
############
# COMPONENTS
############



class VideoEncoder_LSTM(nn.Module):
  def __init__(self, n_features, embedding_dim=64):
    super(VideoEncoder_LSTM, self).__init__()
    # self.seq_len, self.n_features = seq_len, n_features
    # self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
    # self.rnn2 = nn.LSTM(
    #   input_size=self.hidden_dim,
    #   hidden_size=embedding_dim,
    #   num_layers=1,
    #   batch_first=True
    # )

  # def forward(self, x):
  #   x = x.reshape((1, self.seq_len, self.n_features))
  #   x, (_, _) = self.rnn1(x)
  #   x, (hidden_n, _) = self.rnn2(x)
  #   return hidden_n.reshape((self.n_features, self.embedding_dim))


class VideoDecoder_LSTM(nn.Module):
  def __init__(self, input_dim=64, seq_len=1):
    super(VideoDecoder_LSTM, self).__init__()
    # self.seq_len, self.input_dim = seq_len, input_dim
    # self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=seq_len,
      num_layers=1,
      batch_first=True
    )
    # self.rnn2 = nn.LSTM(
    #   input_size=2 * input_dim,
    #   hidden_size=self.hidden_dim,
    #   num_layers=1,
    #   batch_first=True
    # )

    # self.output_layer = nn.Linear(self.hidden_dim, n_features)

  # def forward(self, x):
  #   x = x.repeat(self.seq_len, self.n_features)
  #   x = x.reshape((self.n_features, self.seq_len, self.input_dim))
  #   x, (hidden_n, cell_n) = self.rnn1(x)
  #   x, (hidden_n, cell_n) = self.rnn2(x)
  #   x = x.reshape((self.seq_len, self.hidden_dim))
  #   return self.output_layer(x)

class VideoEncoder_GRU(nn.Module):
  def __init__(self, n_features, embedding_dim=64):
    super(VideoEncoder_GRU, self).__init__()
    # self.seq_len, self.n_features = seq_len, n_features
    # self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.GRU(
      input_size=n_features,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
    # self.rnn2 = nn.LSTM(
    #   input_size=self.hidden_dim,
    #   hidden_size=embedding_dim,
    #   num_layers=1,
    #   batch_first=True
    # )

  # def forward(self, x):
  #   x = x.reshape((1, self.seq_len, self.n_features))
  #   x, (_, _) = self.rnn1(x)
  #   x, (hidden_n, _) = self.rnn2(x)
  #   return hidden_n.reshape((self.n_features, self.embedding_dim))


class VideoDecoder_GRU(nn.Module):
  def __init__(self, input_dim=64, seq_len=1):
    super(VideoDecoder_GRU, self).__init__()
    # self.seq_len, self.input_dim = seq_len, input_dim
    # self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.GRU(
      input_size=input_dim,
      hidden_size=seq_len,
      num_layers=1,
      batch_first=True
    )
    # self.rnn2 = nn.LSTM(
    #   input_size=2 * input_dim,
    #   hidden_size=self.hidden_dim,
    #   num_layers=1,
    #   batch_first=True
    # )

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
        #Now trying With LSTM
        # self.video_encoder = VideoEncoder_LSTM(n_features=32 * 75, embedding_dim=128)
        # self.video_decoder = VideoDecoder_LSTM(input_dim=128, seq_len=batch_size * 75)

        # Now trying With GRU
        self.video_encoder = VideoEncoder_GRU(n_features=32 * 75, embedding_dim=128)
        self.video_decoder = VideoDecoder_GRU(input_dim=128, seq_len=batch_size * 75)

    def forward(self, x):
        #taking input as a video but then pass it as a skeleton

        x = self.transform_input_for_skeleton(x)

        # Forward pass For Skel_AutoEncoder
        skel_encoded = self.skel.encoder(x)
        skel_decoded = self.skel.decoder(skel_encoded)
        # skeleton part finishes

        # Forward pass For Video_AutoEncoder
        # Reshaping Skeleten_ecoded embeddings Accordingly
        input_video_encoder = self.transform_input_for_video(skel_encoded)

        # video_encoded = self.video_encoder.encoder(input_video_encoder)
        input_video_encoder = input_video_encoder.reshape((1,100,2400))
        video_encoded, _ = self.video_encoder.rnn1(input_video_encoder)

        input_video_decoded = video_encoded.reshape((1, 100, 128))
        video_decoded, _ = self.video_decoder.rnn1(input_video_decoded)

        # Reshaping Video Decoded such that skeleton can be reproduced

        video_decoded = self.transform_input_for_skeleton(video_decoded)


        x = self.layer_1(video_encoded.reshape(100, 128))
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