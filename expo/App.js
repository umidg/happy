import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Image,
  Pressable,
  Modal,
  Button,
} from 'react-native';
import { CameraModules } from './src/Camera';
import * as ImagePicker from 'expo-image-picker';
import { IconButton } from 'react-native-paper';
import { api } from './src/api';
import { Video, Audio, AVPlaybackStatus } from 'expo-av';
import { manipulateAsync, FlipType, SaveFormat } from 'expo-image-manipulator';

export default function App() {
  const [openImageCapture, setOpenImageCapture] = useState(false);
  const [image, setImage] = useState('');
  const [result, setResult] = useState(null);
  const video = React.useRef(null);
  const [status, setStatus] = React.useState({});
  const [responseVideo, setResponseVideo] = useState(null);

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      // aspect: [4, 3],
      quality: 0.5,
    });

    console.log(result, 'result');

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  useEffect(() => {
    const enableAudio = async () => {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,

        playsInSilentModeIOS: true,
        staysActiveInBackground: false,

        shouldDuckAndroid: false,
      });
    };
    enableAudio();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View
        style={{
          paddingTop: 20,
        }}
      >
        <Image
          style={{
            width: '100%',
            alignSelf: 'center',
            resizeMode: 'contain',
          }}
          source={require('./assets/happyscribbles.png')}
        />
        <View>
          <Image
            style={{
              alignSelf: 'center',
            }}
            source={require('./assets/beaver.png')}
          />
        </View>
      </View>
      <View
        style={{
          width: '100%',
        }}
      >
        <View>
          <Pressable
            style={styles.box}
            onPress={() => setOpenImageCapture(true)}
          >
            <Text style={styles.text}>Take Snap</Text>
          </Pressable>
        </View>
        <Pressable style={styles.box} onPress={() => pickImage()}>
          <Text style={styles.text}>Camera Roll</Text>
        </Pressable>
      </View>
      {openImageCapture && (
        <CameraModules
          cameraBool={openImageCapture}
          closeCamera={() => setOpenImageCapture(false)}
          imageProp={(uri) => {
            console.log(uri, 'uri');
            setImage(uri);
          }}
        />
      )}

      <Modal
        animationType='slide'
        transparent={false}
        visible={image ? true : false}
      >
        <View
          style={{
            paddingTop: 50,
          }}
        ></View>
        <View style={{}}>
          <Image
            style={{
              width: '100%',
              height: '100%',
            }}
            source={{ uri: image }}
          />
          <IconButton
            icon='close-thick'
            size={40}
            iconColor='white'
            onPress={() => setImage('')}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: 20,
              bottom: 0,
              borderWidth: 1,
              borderColor: 'white',
              elevation: 3,
            }}
          />
          <Pressable
            style={{
              position: 'absolute',
              right: 20,
              bottom: 100,
              borderWidth: 1,
              borderColor: 'white',
              elevation: 3,
              borderRadius: 40,
              backgroundColor: '#A020F0',
            }}
            onPress={async () => {
              console.log(image, 'iamge');
              const videoResponse = await api(image);
              console.log(videoResponse, 'video');
              if (videoResponse?.data) {
                setResponseVideo(videoResponse.data);
                setImage(null);
              }
            }}
          >
            <Text
              style={{
                padding: 15,
                paddingHorizontal: 40,
                color: 'white',
                fontSize: 24,
                fontWeight: '700',
              }}
            >
              Send
            </Text>
          </Pressable>
        </View>
      </Modal>
      <Modal
        animationType='slide'
        transparent={false}
        visible={responseVideo ? true : false}
      >
        <View>
          <Video
            ref={video}
            style={styles.video}
            source={{
              uri: responseVideo,
            }}
            useNativeControls
            resizeMode='contain'
            isLooping
            onPlaybackStatusUpdate={(status) => setStatus(() => status)}
          />
          <View style={styles.buttons}>
            <Button
              title={status.isPlaying ? 'Pause' : 'Play'}
              onPress={() =>
                status.isPlaying
                  ? video.current.pauseAsync()
                  : video.current.playAsync()
              }
            />
            <Button
              title='Close'
              onPress={() => {
                setImage(null);
                setResponseVideo(null);
              }}
            />
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
    margin: 10,
  },
  box: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    width: '100%',
    paddingVertical: 45,
    paddingHorizontal: 25,
    backgroundColor: 'white',
    borderRadius: 5,
    marginBottom: 20,
  },
  text: {
    fontSize: 24,
    fontWeight: '700',
  },
  video: {
    alignSelf: 'center',
    width: '100%',
    height: 600,
  },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
