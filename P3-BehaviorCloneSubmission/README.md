/////////MODEL DESIGN ///////////
This was a very challenging project.  There were times where I was very discouraged and didn't know what to do next, and thought I was really thrown into the deep end.  However, I feel like I learned a lot more having to struggle through this and work with other students on slack and in the forums.

When I first started, I assumed I would need a big elaborate neural network to do such a complicated task.  I think I had 6 or 7 convolution layers with the last one 256 filters deep.  I then had 3 fully connected layers.  This gave a network that was in the millions of parameters.  It didn't work at all though.  After much troubleshooting, I found that the generator and the data was just as critical, if not more, than network complexity.  

I ran a preprocessing algorithm on the image that removed the top 50 pixels and the bottom 20 pixels of the image.  This is to remove information that doesn't help the driving, like the hood on the car and trees and mountains above the roadway.  It then converted the image from RBG to HSV.  The last thing that the preprocessor does is scale the image down to 64x64 pixels.  The network just doesn't need a lot of the details from each picture, so this slims down the parameters in the network, which improves training and execution speed without losing performance.  Naturally, the same preprocessing was used for training, validation and in real time on the track driving simulation.


/////////TRAINING APPROACH/////

My test data consisted of driving samples I collected myself in the training program.  I own a video game controller, so I had smooth turn values through all the corners.  I recorded several laps around the track normally, then I supplemented the high angle curve data by going around the curves after the bridge in both directions a few times.  I also included data from a single drive through track 2, in attempt to prevent over-fitting.  My data set had approximately 4000 samples and I split 10% off for validation data.  I didn't create a test data split, because the track performance generates test data as it runs.

I used Keras, and my model trained on the fit_generator function.  Therefor, I had to build a generator.

I created a generator that used a list based on a CSV file reader as the index for all the images and took parameters for batch size and a boolean to select whether the generator should give training or validation data.  The generator would do the normal processing above, but it had a couple additional steps for training.  

The first modification is that it would randomly select middle, right or left camera.  If the left or right cameras were selected, an offset value of +.25 or -.25 was applied to command the vehicle towards the center of the road/data.  All of my training data was ideal driving; there was no data for recovering from being on the line or anything.

The next step was to select a line from the CSV file to use.  To accomplish this, I applied a minimum threshold for the turn angle for the selection.  Basically, this forced the system to learn turns greater than the values specified, and the generator would keep picking random lines until it found one that exceeded the angle threshold.  By doing this, I could control the distribution of training data.  I only needed minimum thresholding because the steering angle data is highly 0 centered, so I really just needed to force it to pick data that had higher angles.  To select the threshold, I had a numpy.random.select on an array of [0,.03,.03,.05,.3,.35].  Since it should pick each one approximately equal times, this translates to 1/6 of selections can be 0 angle, 1/3 is greater than .03, 1/6 greater than .05...etc...  This control over the angles that were available to learn on became essential to the behavior of the network, and it's also how I got it to learn high angle turns.

The next thing the learning generator did was flip every other picture.  Most of the data goes to left turns, so this just flipped half the data, and reversed the steering angle.  This inevitably flipped some right turn data, so I hope this helped reduce over-fitting and generalize the concept of turning.

The last thing the generator did was add noise to the steer angle.  I found that this was also critical to prevent over-fitting.  All it did was add a random value in the range of +/-.02 units.  This wasn't enough to change the command of an image completely, but I trained two networks back to back where the first was +/-.01 and the second was +/-.02, and the performance on the second network was much better.

For training, I used an Adam optimizer with default learning parameters.  I used a batch size of 256.  I trained for a single epoch with 40,000 samples.  I used the above generator for training, and a seperate generator based on the validation data that would return just the preprocessed image and actual steering angle.  I mention in the lessons learned, that I tried a lot of different training epochs and sizes before realizing I was just causing the network to overfit.  The larger batch size (I started at 64) also seemed to help with training time, so running for more epochs didn't always result in improved validation loss.


////TRAINING LESSONS LEARNED///

Lesson 1: I assumed network complexity and size was good/necessary!

After trying a lot of different network shapes and sizes with little improvement, I saw some work on slack where extremely small networks were built that work very well.  Based on that, I built a relatively small network and focused closer on training data.  I think I originally had a 10 layer network with millions of parameters.

Lesson 2: Always create validation sets!

I wasted a lot of time by not having validation data set aside.  I easily spent 20 or 30 hours on this project and I got a working solution.  I would train, then run predict on a couple images just to make sure it was responsive, then I would watch it on the track.  My mistake was that the track was a good validator, when it doesn't actually tell you that much.
I went back and looked at the rubric, requiring a training/validation data split.  After I added that, a lot of the issues I had been battling become much clearer.  I quickly realized my network was over-fitting, badly, which I didn't really even realize until the history file showed me my training loss was going down and my validation loss was increasing.  I repeated the activity of training the network with this information, and I went from training for 10-20 epochs with 40k samples, down to a single epoch.  I made this choice based on the validation loss going up instead of coming down after several epochs.
This also allowed me to quickly test changes (like adding variation to the steer angle command during training).

Lesson 3: Spend more time reviewing image formats and picture data!
I started trying to use RGB and thought that more picture data had to be a good thing.  After reviewing the data in HSV format, I realized there was much better numerical separation of roadway and surrounding areas.  I also experimented with various images sizes and croppings.  Ultimately, the scaled down image at 64x64 pixels of just the roadway up to the horizon was the best.  Based on others' work I've seen on slack, I suspect I could scale this down more and still get a functioning network.

The picture data and resulution is the biggest surprise to me.  For classification, of course you want more resolution and higher feature counts, but for this regression task, the network really only cares about the roadway and boundaries.  If you think about it, it's only a handful of features it needs to recognize.  I've seen other students strip down the image size to 16x16 in different color spaces and get functional networks.  My hypothesis is that the image processing they're doing is, somewhat coincidentally, similar to a few steps of convultion and maxpooling.

Lesson 4: Control the training data!
The first results of training yielded a car that just wouldn't steer that hard in a turn.  It became clear that most of the data is straight or low angle turns, so the network thought it was ok.  Clear overfitting to the data that was given to it.  As I added features in the generator to force it to look at more turns than straight roads, the network behavior improved and I could teach it to really turn much harder in corners.



////////MODEL DESCRIPTION////////
The first layer is a normalization, which is lamda x: x/127.5-1.  It preserves the 3 dimensions of the image.

In the following discussion of layers, I don't include the normalization layer above, since this is typical preprocessing.

Every layer after this includes relu activation to introduce non-linearity.

The next layer(1) is a 1x1x3 Convolution layer.  This is meant to allow the network to use the HSV layer with the best information.

Layer 2 and 3 are the same.  They are meant to extract features and reduce dimensionality.  
They are 3x3 convolution with 32 filters.  They each have 2x2 maxpooling applied to quickly get features down into a feature map with a lower number of parameters/dimensions.  Dropout is used to encourage redundancy and prevent overfitting.

Layer 4 is the final convolution layer and is 3x3x64.  This is also followed by maxpooling and dropout.

I found that maxpooling and dropout after each of the above 3 layers was necessary to prevent overfitting and get the size of the network down.

After Convolution 4, the network is flattened and sent to a small fully connected layer.
Layer 5 is the fully connected layer and has 20 neurons.

Layer 6 is the single output which is the calculated steering angle.

////////KERAS MODEL SUMMARY//////
___________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 3)     12          lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 62, 62, 32)    896         convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 31, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 31, 32)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 29, 29, 32)    9248        dropout_1[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 14, 14, 32)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 14, 32)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 12, 12, 64)    18496       dropout_2[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 6, 6, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 6, 6, 64)      0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2304)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 20)            46100       flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 20)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             21          dropout_4[0][0]                  
====================================================================================================
Total params: 74,773
Trainable params: 74,773
Non-trainable params: 0
