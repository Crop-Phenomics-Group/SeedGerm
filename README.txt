SeedGerm is a platform designed to take an RGB image series of a germination experiment as input and
proceed to classify each seed in every image as germinated or non-germinated. Morphological traits of
all seeds are recorded throughout the experiment.

*****     RUNNING SEEDGERM     *****

*Via source code*
1. clone SeedGerm source code repository or download zip file containing source code
2. open command prompt and change directory to the folder containing main.py
3. enter the following command: 'python main.py'

*Via .exe file*
1. download .exe file from SeedGerm's release link
2. double click the SeedGerm.exe file or open a cmd in the .exe file's directory and enter the following command: 'SeedGerm.exe'

*****     USING SEEDGERM     *****

1. Adding an experiment

To create a new experiment, click 'File' in the top left, then 'Add experiment'. This will bring up a window containing 
experiment parameters to be set:
Experiment name - the name you want to assign to this experiment
Image directory - the folder containing the image series to be analysed
Species - the species of the crop found in the image series
Number of panels - the number of separated regions in each image that contain seeds,
		     if only one distinct background/region containing seeds exists, choose 1
Rows - the number of rows of seeds
Cols - the number of columns of seeds
Start image - the index in the image series that you would like the experiment to start at, leaving this empty results in 
              starting at the first image in the series
End image - the index in the image series that you would like the experiment to end at, leaving this empty results in starting
            at the last image in the series
BG remover - the machine learning algorithm used to segment the seeds, SGD can be used in most situations, our paper contains 
             more details regarding when other algorithms may be appropriate
Use Colour - check the box if you want colour features to be used for classifying the seed, particularly useful when the radicle
             is a different colour to the seed coat

Once all parameters have been set, click 'Add' to finish creating your experiment.

2a. Setting YUV thresholds

To train the chosen BG remover algorithm to segment the seeds, YUV thresholds are chosen to create labelled data. To set the YUV 
thresholds, right click on an experiment and click 'Set YUV ranges'.

This will open a new window containing four images and six slider bars, each representing the lower or upper thresholds of the 
Y, U, V colour channels. Usually, the seeds can be successfully segmented by only changing the 'Low' bar of the 'V' colour 
channel. Adjust the thresholds until all seeds in all images are black with the background's colour not changing. Once the 
thresholds yield a desirable result, press 'Enter' on your keyboard to confirm these threshold values.

2b. Setting YUV panel thresholds (Optional)

If the conditions across the panels vary too much (e.g. brightness/contrast), YUV thresholds can be set for each individual 
panel. To set panel-specific thresholds, right click on an experiment and click 'Set YUV_panel ranges'.

After a short delay, this will open a new window containing four images and six slider bars. The objective of this stage is the
same as 2a (all seeds black, background's colour not changing). Once the thresholds yield a desirable result, press 'Enter' on 
your keyboard to confirm these threshold values. This process will be repeated for each panel.

3. Process images

Once YUV thresholds have been set for an image series, it can be processed. To do this, right click on an experiment and click 
'Process images'. The core part of the algorithm 
will run that does the following:
- Segments the individual panels in all images
- Segments the seeds in all images
- Extracts the morphological traits from all seeds in all images
- Classifies all seeds in all images as either germinated or non-germinated
- Presents experiment results graphically (germination uniformity box-plot, germination rate bar chart, cumulative germination 
  line graph)
- Saves cumulative germination counts as csv (panel_cumulative_germ.csv), morphological traits of seeds over experiment as csv 
  (stats_over_time.csv)

4. View experiment results/images/masks (optional)

To view the results of an experiment, right click on said experiment in the main window and click 'View results'. This will 
display the graphical experiment results.
To view the image series of an experiment, right click on said experiment in the main window and click 'View images'.
To view the seed masks used for segmentation in an experiment, right click on an experiment in the main window and click 
'View seed masks'. In the window containing the seed masks, clicking 'start' will show a time lapse of the seed masks across 
the experiment. Red dots denote that a seed that has been classified as germinated.

5. Deleting an experiment

To delete an experiment, right click on said experiment in the main window and select 'Delete'. This will delete everything
associated with this experiment including all csv files, saved image masks, mask images, etc.

6. Editing an experiment's parameters

To edit experimental parameters, right click on an experiment in the main window and select 'Edit'. This will open a window 
identical to the window used for adding an experiment. Modify any experimental parameters and then click 'Confirm' to save 
any changes made to the experimental parameters.
