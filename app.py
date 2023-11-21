from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
# from tqdm import tqdm
# import random
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
import mediapipe as mp
from flask_cors import CORS
import cv2
# import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Reshape, Conv1D, BatchNormalization, DepthwiseConv1D, MaxPool1D, GlobalAvgPool1D, Dropout, Dense
from tensorflow.keras import layers, optimizers
from mapping import signs
import tkinter as tk
from PIL import Image, ImageTk


#preprocess input
def scaled_dot_product(q, k, v, softmax):
    # Calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # Calculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax_output = softmax(scaled_qkt)

    z = tf.matmul(softmax_output, v)
    # Shape: (m, Tx, depth), same shape as q, k, v
    return z

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax))
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

class Transformer(tf.keras.Model):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.mhas = []
        self.mlps = []

        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(UNITS, 8))  # Modify the arguments as needed
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(UNITS, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(0.30),
                tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))

    def call(self, x):
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x)
            x = x + mlp(x)

        return x
    


def dense_block(units):
    fc = layers.Dense(units)
    norm = layers.LayerNormalization()
    act = layers.Activation("gelu")
    drop = layers.Dropout(0.05)
    return lambda x: drop(act(norm(fc(x))))

class FeaturePreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in):
        n_frames = x_in.shape[0]

        # Normalization to a common mean
        x_in = x_in - x_in[~torch.isnan(x_in)].mean(0,keepdim=True) 
        x_in = x_in / x_in[~torch.isnan(x_in)].std(0, keepdim=True)

        # Landmarks reduction
        lips     = x_in[:, IDX_MAP['lips']]
        lhand    = x_in[:, IDX_MAP['left_hand']]
        pose     = x_in[:, IDX_MAP['upper_body']]
        rhand    = x_in[:, IDX_MAP['right_hand']]
        x_in = torch.cat([lips,
                          lhand,
                          pose,
                          rhand], 1) # (n_frames, n_landmarks, 3)

        # Replace nan with 0 before Interpolation
        x_in[torch.isnan(x_in)] = 0

        # If n_frames < k, use linear interpolation,
        # else, use nearest neighbor interpolation
        x_in = x_in.permute(2,1,0) #(3, n_landmarks, n_frames)
        if n_frames < FIXED_FRAMES:
            x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode= 'linear')
        else:
            x_in = F.interpolate(x_in, size=(FIXED_FRAMES), mode= 'nearest-exact')

        return x_in.permute(2,1,0) # (n_frames, n_landmarks, 3)
    

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)



UNITS = 256

# Transformer
NUM_BLOCKS = 2
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu

ROWS_PER_FRAME = 543

    # Check if the function returns in required dimensions
# single_pq = load_relevant_data_subset(f'landmarks/{id}.parquet')


FIXED_FRAMES = 30
# Choose required landmarks only, i.e. reduction from 543 to 104
IDX_MAP = {"lips"       : np.array([
                            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,]).tolist(),
        "left_hand"  : np.arange(468, 489).tolist(),
        "upper_body" : np.arange(489, 511).tolist(),
        "right_hand" : np.arange(522, 543).tolist()}





inputs = tf.keras.Input(shape=(30, 104, 3))

lips = tf.slice(inputs, [0, 0, 0, 0], [-1, 30, 40, 3])
lh = tf.slice(inputs, [0, 0, 40, 0], [-1, 30, 21, 3])
po = tf.slice(inputs, [0, 0, 61, 0], [-1, 30, 22, 3])
rh = tf.slice(inputs, [0, 0, 83, 0], [-1, 30, 21, 3])

lips = tf.keras.layers.Reshape((30, 40*3))(lips)
lh = tf.keras.layers.Reshape((30, 21*3))(lh)
po = tf.keras.layers.Reshape((30, 22*3))(po)
rh = tf.keras.layers.Reshape((30, 21*3))(rh)

# print(lips.shape)
#reshape_layer = tf.keras.layers.Reshape((30, 104*3))(inputs)

embedding_units = [256, 256] # tune this

# dense encoder model
lips = Dense(512, activation="gelu")(lips)
lh = Dense(512, activation="gelu")(lh)
po = Dense(512, activation="gelu")(po)
rh = Dense(512, activation="gelu")(rh)

x = tf.concat((lips, lh, po, rh), axis=2)
# x = tf.reduce_mean(x, axis=3)
for n in embedding_units:
    x = dense_block(n)(x)
x = Transformer(num_blocks=4)(x)
x = tf.reduce_sum(x, axis=1)
dense = layers.Dense(256, activation="gelu")(x)
drop = layers.Dropout(0.1)(x)

out = layers.Dense(262, activation="softmax", name="outputs")(x)


model = tf.keras.Model(inputs=inputs, outputs=out)
# model.summary()


model.load_weights('model/model.h5')
print("model loaded")




def load_process_predict(video_path,id):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

    video_file = video_path
    print(video_file)
    cap = cv2.VideoCapture(video_file)

    video_frames = []
    frame_no = 0
    while cap.isOpened():
        
        success, image = cap.read()

        if not success: break
        image = cv2.resize(image, dsize=None, fx=4, fy=4)
        height,width,_ = image.shape

        #print(image.shape)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = holistic.process(image)
        
        data = [] 
        fy = height/width

        if result.face_landmarks is None:
            for i in range(468): #
                data.append({
                    'type' : 'face',
                    'landmark_index' : i,
                    'x' : np.nan,
                    'y' : np.nan,
                    'z' : np.nan,
                })
        else:
            assert(len(result.face_landmarks.landmark)==468)
            for i in range(468): #
                xyz = result.face_landmarks.landmark[i]
                data.append({
                    'type' : 'face',
                    'landmark_index' : i,
                    'x' : xyz.x,
                    'y' : xyz.y *fy,
                    'z' : xyz.z,
                })

        # -----------------------------------------------------
        if result.left_hand_landmarks is None:
            for i in range(21):  #
                data.append({
                    'type': 'left_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.left_hand_landmarks.landmark) == 21)
            for i in range(21):  #
                xyz = result.left_hand_landmarks.landmark[i]
                data.append({
                    'type': 'left_hand',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })

        if result.pose_landmarks is None:
            for i in range(33):  #
                data.append({
                    'type': 'pose',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.pose_landmarks.landmark) == 33)
            for i in range(33):  #
                xyz = result.pose_landmarks.landmark[i]
                data.append({
                    'type': 'pose',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })

        # -----------------------------------------------------
        if result.right_hand_landmarks is None:
            for i in range(21):  #
                data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                })
        else:
            assert (len(result.right_hand_landmarks.landmark) == 21)
            for i in range(21):  #
                xyz = result.right_hand_landmarks.landmark[i]
                data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': xyz.x,
                    'y': xyz.y *fy,
                    'z': xyz.z,
                })
            zz = 0
        frame_df = pd.DataFrame(data)
        frame_df.loc[:,'frame'] =  frame_no
        video_frames.append(frame_df)


        #=========================
        frame_no += 1

    video_df = pd.concat(video_frames, ignore_index=True)
    # video_df_list.append(video_df)
    output_folder = "landmarks"
    os.makedirs(output_folder, exist_ok=True)
    parquet_file_path = os.path.join(output_folder, f"{id}.parquet")
    # os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)  # Create directories if they don't exist
    video_df.to_parquet(parquet_file_path)

    cap.release()
    holistic.close()

    # reading the parquet file and feature generation

    # Load a single file for visualizing
    df = pd.read_parquet(f'landmarks/{id}.parquet')
    df.sample(10)
    # Load parquet file and convert it to required shape
 


        
    x_in = torch.tensor(load_relevant_data_subset(f'landmarks/{id}.parquet'))
    feature_preprocess = FeaturePreprocess()
    print(feature_preprocess(x_in).shape, x_in[0])



    inputX = feature_preprocess(x_in)
    inputX = inputX.cpu().detach().numpy()

    inputX = np.expand_dims(inputX, axis=0)

    preds = model.predict(inputX)

    ind=np.argmax(preds)

    return signs[ind]


# @app.route('/',methods=['GET'])
# def wel():
#     return "mshdvbh"
# @app.route('/upload_video', methods=['POST'])
# def upload_video():
#     try:
#         video_file = request.files['video']

#         # Save the video file to the server (or process it directly)
        
#         unique_number = str(uuid.uuid4())
#         print("Unique Number:", unique_number)
#         video_path = f'uploads/video{unique_number}.mp4'
#         video_file.save(video_path)
#         prediction=load_process_predict(video_path,unique_number)
#         # Process the video file (implement your logic here)
#         print(prediction)
#         return jsonify({'message': 'Video uploaded and processed successfully.',"pred":prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})


class VideoRecorder(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.configure(bg="#F0F0F0")  # Setting overall background color

        self.is_recording = False
        self.recording_count = 0
        self.video_writers = {} 
        
        # Canvas with padding and background color
        self.canvas = tk.Canvas(self, width=640, height=480, bd=5, relief=tk.SOLID, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # Frame for upper buttons
        upper_button_frame = tk.Frame(self, bg="#F0F0F0")
        upper_button_frame.pack()

        # Toggle recording button with color and font style
        self.toggle_button = tk.Button(upper_button_frame, text="Start Recording", command=self.toggle_recording, bg="#4CAF50", fg="white", font=('Arial', 12, 'bold'))
        self.toggle_button.pack(side=tk.LEFT, padx=10, pady=5)


        # Frame for lower buttons
        lower_button_frame = tk.Frame(self, bg="#F0F0F0")
        lower_button_frame.pack()
        
        # Save button to save the recording
        self.save_button = tk.Button(lower_button_frame, text="Save", command=self.save_recording, bg="#2196F3", fg="white", font=('Arial', 12, 'bold'))
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Prediction button with color and font style
        self.prediction_button = tk.Button(lower_button_frame, text="Predict", command=self.predict, bg="#2196F3", fg="white", font=('Arial', 12, 'bold'))
        self.prediction_button.pack(padx=5, pady=5)

        # Text box to display print with color and font style
        self.text_box = tk.Text(self, height=4, width=50, bg="#FFFFFF", fg="#333333", font=('Arial', 10))
        self.text_box.pack(padx=10, pady=10)
        # OpenCV video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 for default camera, change if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.update_camera()

    def update_camera(self):
        ret, image = self.cap.read()
        if ret:
            image.flags.writeable = False
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.flip(processed_image, 1)
            frame = processed_image.copy()  # For recording without landmarks

            result = self.holistic.process(processed_image)

            if result.face_landmarks is not None:
                self.mp_drawing.draw_landmarks(processed_image, result.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
            if result.left_hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(processed_image, result.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if result.right_hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(processed_image, result.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if result.pose_landmarks is not None:
                self.mp_drawing.draw_landmarks(processed_image, result.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(processed_image))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Recording without landmarks
            if self.is_recording:
                if self.recording_count not in self.video_writers:
                    self.video_writers[self.recording_count] = cv2.VideoWriter(f'recording_{self.recording_count}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
                self.video_writers[self.recording_count].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.after(10, self.update_camera)
  

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.toggle_button.config(text="Stop Recording", bg="#FF5722")
        else:
            self.toggle_button.config(text="Start Recording", bg="#4CAF50")

    def save_recording(self):
        # Placeholder for saving the recording action
        if self.recording_count in self.video_writers:
            self.video_writers[self.recording_count].release()
            self.text_box.insert(tk.END, f"Recording {self.recording_count} saved.\n")
            self.recording_count += 1  # Increment recording count for the next session

    def predict(self):
        if ( self.recording_count < 1 ):
            self.text_box.insert(tk.END,'No video Saved for prediction')
            return
        self.text_box.insert(tk.END, 'Prediction for the previously recorded video is in process.....') 
        video_path = f'recording_{self.recording_count-1}.mp4'
        print(video_path)
        prediction=load_process_predict(video_path,self.recording_count-1)
        self.text_box.insert(tk.END, f"Prediction: {prediction}\n")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("ISL Translation")
    app = VideoRecorder(root)
    app.pack()
    root.mainloop()
