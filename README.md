# CleanDishes
The goal of this project was to see how Pix2Pix performs under pressure with low data. This split the project into 3 main chunks, with more on the way as this is becoming a playground to fully interact with this technology. With very little data we can very clearly see overfitting, to such an extreme it's almost relaxing knowing this isn't something impossibly intelligent.

### Cleaning All My Dishes
This dataset is only 18 training pictures and 4 validation pictures. We have it outlining of a plate simply because the training dataset is so small, guessing a plate to always exist will earn a fair chunk of points most of the time. This is a picture the model has never seen before, and so it decided it would simply turn it into something it had seen before. This was trained for about 12 hours, which isn't a massive time, but given its 18 pictures perhaps it is a lot of time.

![aUnseen - 4](https://user-images.githubusercontent.com/58519904/230157264-dcd10a68-7efd-46be-a985-2eabf92720d2.png)

### Cleaning Just Plates
Here we can see the blue plate taking a stand again. We have even fewer pictures, only 5 training pictures, and 2 validation pictures. All of our plates have been full-sized plates except for the one small plate in the validation set. As such, it turned this small plate into a mix of the two big plates and called it a day.

![aUnseen - 2](https://user-images.githubusercontent.com/58519904/230158006-445ad87e-5aad-4cb4-a14c-22bb88bbddf1.png)

However, this is a plate with a set of marks the model has not seen before. It produced an image better than I expected.

![aUnseen - 1](https://user-images.githubusercontent.com/58519904/230158230-3e3fb3b5-95ea-4055-8910-f87131dc6b58.png)


### Coloring Anime
This is a large dataset (21Gb) of sketched drawings and the output is that sketch but colored. This model was only trained for half a day, but it can still produce pictures that at least look like the input instead of plates fading into existence. Not all results are great, as the soulless eyes of the 2nd picture can show. I used the anime dataset before moving on to my dataset to confirm that things were working and not simply an "it looks bad because there isn't enough data" issue.

![aUnseen - 101](https://user-images.githubusercontent.com/58519904/230160205-372a20df-7eae-4f63-9b34-d0c0bb26e4f8.png)

![aUnseen - 86](https://user-images.githubusercontent.com/58519904/230158789-fa0574c2-9e49-4952-95f9-4ba00f10f58d.png)
