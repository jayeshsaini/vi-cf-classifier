import tensorflow as tf
import base64

def predict(image_string):

    # Read the image_data
    imgdata = base64.b64decode(image_string)
    with open('image.jpg', 'wb') as f:
        f.write(imgdata)

    image_data = tf.gfile.FastGFile('image.jpg', 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                    in tf.gfile.GFile("logs/trained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print(top_k)
        result = label_lines[top_k[0]]
        return(result)