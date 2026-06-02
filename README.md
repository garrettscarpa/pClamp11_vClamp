This script is for analyzing voltage clamp recordings collected in gapfree mode using pClamp11. The data analyze the first 60 seconds of recording, but this duration is easily adjusted at the top of the script (must be separately adjusted for each script)

View the sample data in the 'Voltage Clamp' folder to see how to structure your data. Briefly, put the gapfree recordings in separate folders winthin one parent folder, where each subfolder title is the condition of it's contents. 

The 2 main scripts are v_clamp_analysis.py and v_clamp_results.py. The others can basically be ignored, as they are for specific case uses. 

Once the data are correctly organized, change the root value at the top of the script to point to your parent directory (that holds the subfolders and data) and run the script (after reviewing the detection metrics at the top).

Use the left and right arrows to cycle through all of the detected peaks. Peaks are accepted by default, but you can reject them with the down arrow on the keyboard. You can re-accept a rejected peak with the up arrow on the keyboard. You can drag the bases to adjust them if they don't proerly align with your event.

<img width="1892" height="957" alt="Screenshot 2026-06-02 at 11 53 30" src="https://github.com/user-attachments/assets/dbb44d68-7170-426d-b3f8-5cfd80ea9edf" />

If you see a peak in the bottom window that wasn't detected, simply click the 'Add Peak' button and then click the peak on the bottom window for it to be automatically added. 

If you need to jump to a specific peak, enter that value in the Go To # window and click enter. MAKE SURE YOU CLICK Q WHEN DONE TO SAVE THE DATA!!!!!!

Once the detection and adjustments are complete, simply run the v_clamp_results.py file to see automatically generated comparison graphs. If you want to separate into populations based on half-width (to separate duplicate events or gap-junction-like events), change switch_by_halfwidth from True to False and set the halfwidth_threshold_ms value.

The script generates 2 figures. The first figure shows the comparison features between the conditions, along with overlays of all events for each condition. 

<img width="1845" height="950" alt="Screenshot 2026-06-02 at 12 04 06" src="https://github.com/user-attachments/assets/ba801603-38e0-470e-ac33-5862c1e88196" />

The second is a validation figure, which shows the sum of the onset and offset times for each condition (time to peak and time to recover). It compares this value to the difference in base values (on the x-axis), and plots the residual (which should be 0). 

<img width="1034" height="538" alt="Screenshot 2026-06-02 at 12 04 53" src="https://github.com/user-attachments/assets/dd1af2a4-78b7-4224-bf7f-f477c36647cf" />

Note that if split_by_halfwidth = True, then 2x the graphs will be generated (one set of graphs for each group of events).
