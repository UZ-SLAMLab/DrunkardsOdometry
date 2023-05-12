#!/bin/bash

folder="/cluster/project/cvg/drecasens/drunk_dataset"
folder_videos="/cluster/project/cvg/drecasens/videos"

scenes="00014"

levels="3"
files="color"

for scene in $scenes
do
   for level_idx in $levels
   do
      level="level$level_idx"
      for file in $files
      do
         echo $folder/$scene/$level/$file

         if [ ! -d $folder_videos/$scene/$level/$file ]; then
               mkdir -p $folder_videos/$scene/$level/$file
         fi

         if [ -f $folder/$scene/$level/$file/"0000000000.png" ]; then
            /cluster/home/drecasens/Installations/ffmpeg-git-20220910-amd64-static/ffmpeg -f image2 -pattern_type sequence -start_number 0 -r 30 -i $folder/$scene/$level/$file/%010d.png -s 1920x1920 $folder_videos/$scene/$level/$file/video.avi
         elif [ -f $folder/$scene/$level/$file/$file/"0000000000.png" ]; then
            /cluster/home/drecasens/Installations/ffmpeg-git-20220910-amd64-static/ffmpeg -f image2 -pattern_type sequence -start_number 0 -r 30 -i $folder/$scene/$level/$file/$file/%010d.png -s 1920x1920 $folder_videos/$scene/$level/$file/video.avi
         fi
      done
   done
done

scenes="00015 00016 00018 00019"

levels="0 1 2 3"
files="color"

for scene in $scenes
do
   for level_idx in $levels
   do
      level="level$level_idx"
      for file in $files
      do
         echo $folder/$scene/$level/$file

         if [ ! -d $folder_videos/$scene/$level/$file ]; then
               mkdir -p $folder_videos/$scene/$level/$file
         fi

         if [ -f $folder/$scene/$level/$file/"0000000000.png" ]; then
            /cluster/home/drecasens/Installations/ffmpeg-git-20220910-amd64-static/ffmpeg -f image2 -pattern_type sequence -start_number 0 -r 30 -i $folder/$scene/$level/$file/%010d.png -s 1920x1920 $folder_videos/$scene/$level/$file/video.avi
         elif [ -f $folder/$scene/$level/$file/$file/"0000000000.png" ]; then
            /cluster/home/drecasens/Installations/ffmpeg-git-20220910-amd64-static/ffmpeg -f image2 -pattern_type sequence -start_number 0 -r 30 -i $folder/$scene/$level/$file/$file/%010d.png -s 1920x1920 $folder_videos/$scene/$level/$file/video.avi
         fi
      done
   done
done

