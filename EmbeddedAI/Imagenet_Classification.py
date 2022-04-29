import tensorflow as tf
import numpy as np
import cv2

def preprocess(image):
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #dst[:,:,0] = cv2.equalizeHist(dst[:,:,0]) # stretching
    img = cv2.resize(dst, (224,224))
    img = img / 255.
    return img
    
def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
    model_path = 'mobilenet_tflite_model.tflite'
    labels = []
    with open('mobilenet_labels.txt', 'r') as f:
        while True:
            line = f.readline()
            if line == "":
                break
            labels.append(line.strip().lower())
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
        
    while(camera.isOpened()):
        keValue = cv2.waitKey(1)
        if keValue == ord('q'):
            break
        
        _, image = camera.read()
        image = cv2.flip(image, -1)
        cv2.imshow('raw', image)
        prep = preprocess(image)
        cv2.imshow('prep', prep)
        input_data = np.array(prep.reshape(input_details[0]['shape']),
                              dtype=input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("Object is ", labels[np.argmax(output_data).squeeze()])
        
    
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()


