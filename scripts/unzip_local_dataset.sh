#!/bin/bash

# Script to unzip the whole Drunkard's Dataset once it has been fully downloaded.

folder="/.../drunkards_dataset"

scenes="00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011 00012 00013 00014 00015 00016 00018 00019"

levels="0 1 2 3"
files="color depth optical_flow normal"

for scene in $scenes
do
   for level_idx in $levels
   do
      level="level$level_idx"
      for file in $files
      do      
         echo $folder/$scene/$level/"$file.zip"
         if [ -f $folder/$scene/$level/"$file.zip" ] && [ ! -d $folder/$scene/$level/$file ]; then
            unzip $folder/$scene/$level/"$file.zip" -d $folder/$scene/$level      
         fi   
      done
   done
done

