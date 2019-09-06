#!/usr/bin/env bash

# download the modified UD v2.2 data
filename=data2.2.zip
fileid="1H8P5xTa1OdVWo2pCVOKzEO-LFimnB9Y4"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip data2.2.zip
rm data2.2.zip

# download the revised embedding files for 31 languages
filename=ud2_embeddings.zip
fileid="1DiH0y-R00v1VsEbHbcBH6_Wg9rjsHz8u"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ud2_embeddings.zip
rm ud2_embeddings.zip

rm ./cookie
