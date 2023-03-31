import React, { useState, useEffect } from 'react';
import { Modal, Text, TouchableOpacity, View, Image } from 'react-native';
import { Camera } from 'expo-camera';
import { Button } from 'react-native-paper';
const CameraModule = (props) => {
  const [cameraRef, setCameraRef] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  return (
    <Modal
      animationType='slide'
      transparent={true}
      visible={true}
      onRequestClose={() => {
        props.setModalVisible();
      }}
    >
      <Camera
        style={{ flex: 1 }}
        ratio='16:9'
        flashMode={Camera.Constants.FlashMode.on}
        type={type}
        ref={(ref) => {
          setCameraRef(ref);
        }}
      >
        <View
          style={{
            flex: 1,
            backgroundColor: 'transparent',
            justifyContent: 'flex-end',
          }}
        >
          <View
            style={{
              backgroundColor: 'black',
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Button
              icon='close'
              style={{ marginLeft: 12 }}
              mode='outlined'
              color='white'
              onPress={() => {
                props.setModalVisible();
              }}
            >
              Close
            </Button>
            <TouchableOpacity
              onPress={async () => {
                if (cameraRef) {
                  let photo = await cameraRef.takePictureAsync();
                  props.setImage(photo);
                  props.setModalVisible();
                }
              }}
            >
              <View
                style={{
                  borderWidth: 2,
                  borderRadius: 50,
                  borderColor: 'white',
                  height: 50,
                  width: 50,
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  marginBottom: 16,
                  marginTop: 16,
                }}
              >
                <View
                  style={{
                    borderWidth: 2,
                    borderRadius: 50,
                    borderColor: 'white',
                    height: 40,
                    width: 40,
                    backgroundColor: 'white',
                  }}
                ></View>
              </View>
            </TouchableOpacity>
            <Button
              icon='axis-z-rotate-clockwise'
              style={{ marginRight: 12 }}
              mode='outlined'
              color='white'
              onPress={() => {
                setType(
                  type === Camera.Constants.Type.back
                    ? Camera.Constants.Type.front
                    : Camera.Constants.Type.back
                );
              }}
            >
              {type === Camera.Constants.Type.back ? 'Front' : 'Back '}
            </Button>
          </View>
        </View>
      </Camera>
    </Modal>
  );
};
export const CameraModules = ({ cameraBool, closeCamera, imageProp }) => {
  const [image, setImage] = useState(null);
  const [camera, setShowCamera] = useState(cameraBool);
  const [hasPermission, setHasPermission] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);
  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      {/* <View
        style={{
          backgroundColor: '#eeee',
          width: 120,
          height: 120,
          borderRadius: 100,
          marginBottom: 8,
        }}
      >
        <Image
          source={{ uri: image }}
          style={{ width: 120, height: 120, borderRadius: 100 }}
        />
      </View> */}
      {/* <Button
        style={{ width: '30%', marginTop: 16 }}
        icon='camera'
        mode='contained'
        onPress={() => {
          setShowCamera(true);
        }}
      >
        Camera
      </Button> */}
      {camera && (
        <CameraModule
          showModal={camera}
          setModalVisible={() => {
            setShowCamera(false);
            closeCamera(false);
          }}
          setImage={(result) => {
            setImage(result.uri);
            imageProp(result.uri);
          }}
        />
      )}
    </View>
  );
};
// App.js
// import React, { useState } from 'react';
// import { View, Text, StyleSheet, Image, Button } from 'react-native';

// import * as ImagePicker from 'expo-image-picker';

// function ImagePickerModule() {
//   // The path of the picked image
//   const [pickedImagePath, setPickedImagePath] = useState('');

//   // This function is triggered when the "Select an image" button pressed
//   const showImagePicker = async () => {
//     // Ask the user for the permission to access the media library
//     const permissionResult =
//       await ImagePicker.requestMediaLibraryPermissionsAsync();

//     if (permissionResult.granted === false) {
//       alert("You've refused to allow this appp to access your photos!");
//       return;
//     }

//     const result = await ImagePicker.launchImageLibraryAsync();

//     // Explore the result
//     console.log(result);

//     if (!result.cancelled) {
//       setPickedImagePath(result.uri);
//       console.log(result.uri);
//     }
//   };

//   // This function is triggered when the "Open camera" button pressed
//   const openCamera = async () => {
//     // Ask the user for the permission to access the camera
//     const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

//     if (permissionResult.granted === false) {
//       alert("You've refused to allow this appp to access your camera!");
//       return;
//     }

//     const result = await ImagePicker.launchCameraAsync();

//     // Explore the result
//     console.log(result);

//     if (!result.cancelled) {
//       setPickedImagePath(result.uri);
//       console.log(result.uri);
//     }
//   };

//   return (
//     <View style={styles.screen}>
//       <View style={styles.buttonContainer}>
//         <Button onPress={showImagePicker} title='Select an image' />
//         <Button onPress={openCamera} title='Open camera' />
//       </View>

//       <View style={styles.imageContainer}>
//         {pickedImagePath !== '' && (
//           <Image source={{ uri: pickedImagePath }} style={styles.image} />
//         )}
//       </View>
//     </View>
//   );
// }

// // Kindacode.com
// // Just some styles
// const styles = StyleSheet.create({
//   screen: {
//     flex: 1,
//     justifyContent: 'center',
//     alignItems: 'center',
//   },
//   buttonContainer: {
//     width: 400,
//     flexDirection: 'row',
//     justifyContent: 'space-around',
//   },
//   imageContainer: {
//     padding: 30,
//   },
//   image: {
//     width: 400,
//     height: 300,
//     resizeMode: 'cover',
//   },
// });

// export default ImagePickerModule;
