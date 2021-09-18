# QR-Code-Project
Project based on https://web.stanford.edu/class/ee368/Project_06/project.html, where a MATLAB script tries finding and decoding custom QR code(s) in inputted images. Training images 1 - 12 (and training_ground_truth.mat) are from the link above, while images 13 - 18 were added by Joseph to test the program (in particular the skew correction part of the script). The script successfully found and accurately extracted information out of all tested images.

The script can be broken down into three main phases of operation: thresholding, QR code identification, and QR code data extraction.

In thresholding, various image manipulation techniques (e.g. sharpening edges, removing blobs of certain sizing, etc.) are applied to make potential QR codes pop out as blobs on a black background.

These blobs are then examined to see if they're similar enough to a pair of identification bars in the custom QR code. Checking a pair of blobs for a certain orthogonality, distance from each other (for their given size), and ratio of size with respect to each other (given a specified tolerance) are used. Should a pair of blobs pass this test, a sub-image around the blobs is cut out, cleaned up, shear-corrected, and rotated for ease of processing in the next step.

Additional cleanup is applied before the script attempts to find markers in each corner of the QR code. Knowing where each corner is, the script can change where it should look to identify QR code pixels within the area between all the corners (e.g. if the QR has an isosceles trapezoid due to the camera angle the image was taken at, the script would know pixels are more tightly spaced at the top of the QR code compared to the bottom, and each row of QR code changes slighly moving down - provided it correctly identifies the corner markers). The data extracted from each QR code is then translated into a string to obtain the original message.

Note: the loop that iterates through test images was modified from evaluate.m in above link.
