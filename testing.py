import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

COLOR = "yellow"
INPUT_PATH = "images/ground_truth_" + COLOR

blurred_accuracies = []
rgb_accuracies = []
x = []
x_ind = 0

for filename in os.listdir(INPUT_PATH):
    
    f = os.path.join(INPUT_PATH, filename)
    
    s = filename.split("_")
    s2 = s[1].split(".")
    
    img_index = int(s2[0])
    #input image
    ground_truth = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
    rgb_mask = cv2.imread("images/results_"+COLOR+"/masks/combined_rgb/img_"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)
    blurred_mask = cv2.imread("images/results_"+COLOR+"/masks/combined/img_"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)

    
    total_white = np.count_nonzero(ground_truth)
    total_rgb_mask = np.count_nonzero(rgb_mask)
    total_blurred_mask = np.count_nonzero(blurred_mask)
    
    blurred_accuracies.append((total_blurred_mask/total_white)*100)
    rgb_accuracies.append((total_rgb_mask/total_white)*100)
    
    x.append(x_ind)
    x_ind = x_ind + 1
    
average_rgb_accuracy = 0
average_blur_accuracy = 0
average_grey_accuracy = 0

for i, val in enumerate(blurred_accuracies):
    average_blur_accuracy += val
    average_rgb_accuracy += rgb_accuracies[i]
    
average_rgb_accuracy = average_rgb_accuracy/len(rgb_accuracies)
average_blur_accuracy = average_blur_accuracy/len(blurred_accuracies)

print("for color: "+COLOR+" blur accuracy: ",average_blur_accuracy, "\trgb accuracy: ", average_rgb_accuracy)

plt.plot(x, rgb_accuracies, label="rgb", color=COLOR)
plt.plot(x, blurred_accuracies, label="blur", color="black")

plt.xlabel("images")
plt.ylabel('accuracy')
plt.legend()
plt.show()
# plt.savefig("accuracies_"+COLOR+".png")
