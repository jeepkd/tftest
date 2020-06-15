#%%
import tensorflow as tf

#%%
model = tf.keras.applications.DenseNet121(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000
)


# %%
model.summary()

#%%
export_path = 'saved_models/densenet121/1'

# %%
model.save(export_path)

#%%
signatures = {
    'serving_default': model.serve.get_concrete_function(),
}
# options = tf.saved_model.SaveOptions(function_aliases={
#     'my_func': func,
# })
tf.saved_model.save(
    model, export_path, signatures=signatures
)


# %%
import requests
import cv2
import numpy as np
from PIL import Image
import base64

url = 'https://miro.medium.com/max/1400/0*_rx-A_h_zrfH1PyB.jpg'
response = requests.get(url, stream=True)
jpg_buffer = response.content
image = np.frombuffer(jpg_buffer, dtype='int8')
image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
Image.fromarray(image)

#%%
url = 'https://img.kapook.com/u/2016/thachapol/zzzz999999999/24.jpg'
file_name = 'image.jpg'
file_path = tf.keras.utils.get_file(file_name, url)
img = tf.keras.preprocessing.image.load_img(file_path, target_size=[224,224])
img

#%%
x = np.array(img)
x =np.expand_dims(x, 0)
x.shape
y = model.predict(x)

#%%
np.argmax(y)

# %%
import json
# jpeg_bytes = base64.b64encode(jpg_buffer)
image_bytes = base64.b64encode(image)
data={
    'instances': [{'input_2': {'b64': str(image_bytes)}}]
}
predict_request = json.dumps(data)
server_url = 'http://localhost:8501/v1/models/densenet121/versions/1:predict'
response = requests.post(server_url, json=data)
response.content


# %%
response.content.decode()

# %%
