import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

blurred_accuracies = []
rgb_accuracies = []
rgb_to_grey_accuracies = []

x = []
x_ind = 0
for filename in os.listdir("captured_images_blue"):
    
    f = os.path.join('captured_images_blue', filename)
    
    s = filename.split("_")
    s2 = s[1].split(".")
    
    img_index = int(s2[0])
    #input image
    img = cv2.imread(f)
    
    ground_truth = cv2.imread("ground_truth_blue/img"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)
    rgb_mask = cv2.imread("results_blue/masks/combined_rgb/img"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)
    blurred_mask = cv2.imread("results_blue/masks/combined/img_"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)
    rgb_to_grey_mask = cv2.imread("results_blue/masks/rgb_to_grey/img"+str(img_index)+".png", cv2.IMREAD_GRAYSCALE)
    
    
    total_white = np.count_nonzero(ground_truth)
    total_rgb_mask = np.count_nonzero(rgb_mask)
    total_blurred_mask = np.count_nonzero(blurred_mask)
    total_rgb_to_grey = np.count_nonzero(rgb_to_grey_mask)
    
    blurred_accuracies.append((total_blurred_mask/total_white)*100)
    rgb_accuracies.append((total_rgb_mask/total_white)*100)
    rgb_to_grey_accuracies.append((total_rgb_to_grey/total_white)*100)
    
    x.append(x_ind)
    x_ind = x_ind + 1
    
average_rgb_accuracy = 0
average_blur_accuracy = 0
average_grey_accuracy = 0


for i, val in enumerate(blurred_accuracies):
    average_blur_accuracy += val
    average_rgb_accuracy += rgb_accuracies[i]
    average_grey_accuracy += rgb_to_grey_accuracies[i]
    
average_rgb_accuracy = average_rgb_accuracy/len(rgb_accuracies)
average_blur_accuracy = average_blur_accuracy/len(blurred_accuracies)
average_grey_accuracy = average_grey_accuracy/len(rgb_to_grey_accuracies)

print(average_blur_accuracy, "\t", average_rgb_accuracy, "\t", average_grey_accuracy)
# plt.plot(x, blurred_accuracies, label="blurred")
plt.plot(x, rgb_accuracies, label="rgb", color='blue')
# plt.plot(x, rgb_to_grey_accuracies, label="grey", color="grey")

plt.xlabel("images")
plt.ylabel('accuracy')
plt.title("blue cubes")
plt.legend()
plt.show()
plt.savefig("accuracies_all_blue.png")