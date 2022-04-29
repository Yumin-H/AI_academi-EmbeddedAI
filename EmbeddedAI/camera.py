import cv2
import numpy as np

def preprocess(image):
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    #print(np.min(dst[:,:,0]), np.max(dst[:,:,0]))
    #dst[:,:,0] = cv2.equalizeHist(dst[:,:,0]) # stretching
    dst[:,:,0] = cv2.add(dst[:,:,0], 80) # brightness control
    gmin, gmax = np.min(dst[:,:,0]), np.max(dst[:,:,0])
    dst[:,:,0] = (((dst[:,:,0] - gmin)*255.) / (gmax-gmin)).astype(np.uint8)
    dst = cv2.cvtColor(dst, cv2.COLOR_YCR_CB2BGR)
    return dst
    
def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
    pre_image = None
    
    while(camera.isOpened()):
        keValue = cv2.waitKey(1)
        if keValue == ord('q'):
            break
        
        _, image = camera.read()
        image = cv2.flip(image, -1)
        if pre_image is None:
            pre_image = image.copy()
            continue
        else:
            pre_hist = cv2.calcHist([pre_image],[0],None,[256],[0,256])
            hist = cv2.calcHist([image],[0],None,[256],[0,256])
            mse = np.sqrt(np.mean((pre_hist-hist)**2))
            if mse > 1000:
                print("detect~!!")
            else:
                pass
            pre_image = image.copy()
                
        cv2.imshow('raw', image)
        prep = preprocess(image)
        cv2.imshow('prep', prep)
    
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()