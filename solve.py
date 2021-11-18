from re import I
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import pickle
import numpy
import imutils
import cv2


#load model
model = load_model("model_final.hdf5")

#translates predictions to letters
with open("model_labels.dat", "rb") as dict:
    dict1 = pickle.load(dict)

#load images from testing folder
image_folder = list(paths.list_images("testing"))

for base_image in image_folder:
    
    #load image and convert to greyscale
    image = cv2.imread(base_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #add padding
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    #converting to black and white
    bandwhite = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #get contours
    contours = cv2.findContours(bandwhite.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    #make array to store bounds for every charachter
    charachter_bounds = []

    for contour in contours:
        #get bounds
        (x, y, w, h) = cv2.boundingRect(contour)

        #check to see if it is a chunk
        if w / h > 1.30:
            #split in half
            new_width = int(w / 2)
            charachter_bounds.append((x, y, new_width, h))
            charachter_bounds.append((x + new_width, y, new_width, h))
        else:
            charachter_bounds.append((x, y, w, h))


    #sorts images based on x bound
    charachter_bounds = sorted(charachter_bounds, key=lambda x: x[0])

    #save output image
    output = cv2.merge([image] * 3)
    predictions = []

    #loop over bounds and create letter images
    for bounds in charachter_bounds:

        x, y, w, h = bounds
        indiv_letter = image[y - 2:y + h + 2, x - 2:x + w + 2]
        indiv_letter = resize_to_fit(indiv_letter, 20, 20)
        indiv_letter = numpy.expand_dims(indiv_letter, axis=2)
        indiv_letter = numpy.expand_dims(indiv_letter, axis=0)

        #send letter into nueral network
        output_nn = model.predict(indiv_letter)
        
        #get nueral network prediction from output
        charachter = dict1.inverse_transform(output_nn)[0]
        predictions.append(charachter)

        # write prediction on image
        cv2.putText(output, charachter, (x - 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    # Print the captcha's text
    output_text = ""
    for i in predictions:
        output_text = output_text + i   
    print("Solved Captcha: " + output_text)

    #display image
    cv2.imshow("Output", output)
    cv2.waitKey()