Image Segmentaion and Classifier do not need to be run for the solution to work they mearly exist to showcase what the models look like.

Fix PapilaDB is a long python file geared to fix the data set i used for training e.g. normalise data so on soforth.

Solution.py is a simple python GUI program the user can click a button it then allows them to select a image from their computer
the program then scales the image down to 256 by 256 while keepiung the aspect ratio consistent feeds it into the Feature exstractor model
then it obtains the shape of the optic cup and disc from the image give that data to the Classifier model and return a diagnosis