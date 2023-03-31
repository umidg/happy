import React from 'react';
import axios from 'axios';
import { manipulateAsync, FlipType, SaveFormat } from 'expo-image-manipulator';

export const api = async (image) => {
  let height = 0;
  let width = 0;

  const formData = new FormData();
  // console.log(image, image.split('.')[1], '\n');
  const gg = Platform.OS === 'ios' ? image.replace('file://', '') : image;
  let manipResult = await manipulateAsync(
    image,
    [
      { rotate: 0 },
      // {
      //   resize: {
      //     width: 500,
      //     height: 500,
      //   },
      // },
    ],
    {
      compress: 0.5,
      format: SaveFormat.PNG,
    }
  ).catch((e) => console.log(e));

  const value = 1000;

  if (manipResult.height > value || manipResult.width > value) {
    if (manipResult.height > manipResult.width) {
      width = manipResult.width / (manipResult.height / value);
      height = value || 500;
    } else {
      height = manipResult.height / (manipResult.width / value);
      width = value || 500;
    }
  }

  console.log(manipResult, 'mani');

  manipResult = await manipulateAsync(
    manipResult.uri,
    [
      { rotate: 0 },
      {
        resize: {
          width: width || 500,
          height: height || 500,
        },
      },
    ],
    {
      format: SaveFormat.PNG,
    }
  ).catch((e) => console.log(e));

  console.log(manipResult, 'mani');

  const img =
    Platform.OS === 'ios'
      ? manipResult.uri.replace('file://', '')
      : manipResult.uri;
  const type = manipResult.uri.split('.').pop();

  formData.append('uploaded_file', {
    name: 'image.' + type,
    uri: img,
    type: 'image/png',
  });

  return axios
    .post('http://157.230.226.238/images', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }) //157.230.226.238
    .then((data) => {
      return data;
    })
    .catch((e) => console.log(JSON.stringify(e)));
};
