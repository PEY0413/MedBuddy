#!/usr/bin/env python

"""
	To run this script, you need to run the below lines in the terminal:
	-$ roslaunch usb_cam usb_cam-test.launch
	-$ roslaunch opencv_apps face_detection.launch image:=/usb_cam/image_raw
	-$ rosrun rchomeedu_vision take_photo_sub.py
	-$ rosrun rchomeedu_speech google_sr.py
	-$ rosrun med_buddy med_buddy.py

	OR

	-$ roslaunch med_buddy med_buddy.launch
"""

import rospy
from sound_play.libsoundplay import SoundClient
from opencv_apps.msg import FaceArrayStamped
from std_msgs.msg import String
from gtts import gTTS
import pandas as pd
import os
from utils import preprocess_text, tfidf, getDisease, getAction, validate_input

# Load data from CSV files
data1 = pd.read_csv('/home/mustar/catkin_ws/src/med_buddy/data/Symptom2Disease.csv') # Symptoms to disease dataset 
data2 = pd.read_csv('/home/mustar/catkin_ws/src/med_buddy/data/Disease2Action.csv') # Disease to actions dataset
symptoms = data1['text']

class MedBuddy:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('med_buddy')
        rospy.sleep(1)  # Allow some time for initialization

        rospy.loginfo("Medbuddy node initialized and ready.")
        
        # Subscribe to face detection topic
        self.face_subscriber = rospy.Subscriber('/face_detection/faces', FaceArrayStamped, self.face_callback)
        rospy.loginfo("Subscribed to /face_detection/faces topic")
        
        self.speech_subscriber = None  # Initialize speech subscriber as None
        self.greeted = False  # Flag to check if the robot has greeted the user
        self.speaking = False  # Flag to check if the robot is currently speaking

    def subscribe_to_speech(self):
        # Subscribe to the speech recognition result topic if not already subscribed and not speaking. 
        if self.speech_subscriber is None:
            if not self.speaking:
                self.speech_subscriber = rospy.Subscriber("result", String, self.talkback)
                rospy.loginfo("Subscribed to /result topic")

    def unsubscribe_from_speech(self):
        # Unsubscribe from the speech recognition result topic if currently subscribed.
        if self.speech_subscriber is not None:
            self.speech_subscriber.unregister()
            self.speech_subscriber = None
            rospy.loginfo("Unsubscribed from /result topic")

    def face_callback(self, msg):
        # Callback function for face detection.
        if not self.speaking:  # Only proceed if the robot is not speaking
            if len(msg.faces) > 0:  # Check if any faces are detected
                if not self.greeted:  # Proceed if the robot has not greeted the user
                    faces_data = [face.face for face in msg.faces]
                    eyes_data = [face.eyes if face.eyes else "null" for face in msg.faces]
                    
                    rospy.loginfo(eyes_data)  # Log detected eyes data
                    if any(eyes != "null" for eyes in eyes_data):  # Check if eyes are detected
                        self.speak("Hello, I'm your health buddy. How can I help you?")
                        self.greeted = True  # Set greeted flag to True
                        rospy.sleep(1)  # Allow some time for the greeting to be heard
                        self.subscribe_to_speech()  # Start listening for speech
            else:
                self.greeted = False  # Reset greeted flag if no faces are detected
                self.unsubscribe_from_speech()  # Stop listening for speech

    def speak(self, data):
        # Convert text to speech and play it. 
        self.unsubscribe_from_speech()  # Unsubscribe from speech topic
        self.speaking = True  # Set the speaking flag to True

        # Generate speech using Google Text-to-Speech (gTTS) and play it
        tts = gTTS(data)
        tts.save("speech.mp3")
        os.system("mpg321 speech.mp3")
        os.remove("speech.mp3")
    
        rospy.sleep(1)  # Allow some time for the speech to be heard
        self.speaking = False  # Reset the speaking flag after speaking
	rospy.sleep(2)  # Wait a bit before resubscribing to avoid picking up its own speech
        self.subscribe_to_speech()  # Resubscribe to speech topic

    def talkback(self, msg):
        # Callback function for handling recognized speech.
        if self.greeted:  # Only proceed if the robot has greeted the user
            rospy.loginfo("Entered talkback callback")

            # Process the recognized speech
            user_input = msg.data
            preprocessed_text = preprocess_text(user_input)  # Preprocess user input text
            preprocessed_symptoms = symptoms.apply(preprocess_text)  # Preprocess symptom list

            # Validate the preprocessed input
            is_valid, error_msg = validate_input(preprocessed_text, preprocessed_symptoms)

            if not is_valid:  # Handle invalid input
                if error_msg == 'Insufficient':
                    self.speak("Sorry, could you please provide more details about your condition?")

                elif 'thank you' in msg.data.lower() or 'thank you very much' in msg.data.lower():
                    self.speak("You're welcome. If you need further assistance, please ask again. Take care and have a great day!")

                elif 'bye' in msg.data.lower():
                    self.speak("Good bye. If you need further assistance, please ask again. Take care and have a great day!")

                elif 'hi' in msg.data.lower() or 'hello' in msg.data.lower() or 'morning' in msg.data.lower():
                    self.speak("Hi! I'm a simple expert system. Feel free to provide your symptoms, and I'll do my best to help you.")

                elif error_msg == 'Not Valid':
                    self.speak("I'm sorry, your input does not contain recognizable symptoms. Please provide valid symptoms.")

                else:
                    self.speak("So sorry, could you please repeat your symptoms?")

            else:  # Handle valid input
                symptom_tfidf = tfidf(preprocessed_text, preprocessed_symptoms)  # Compute TF-IDF for symptoms
                predicted_disease = getDisease(symptom_tfidf)  # Predict disease based on symptoms
                action = getAction(data2, predicted_disease)  # Get action recommendations for the predicted disease
                self.speak("I predict that your disease might be ")
                self.speak(predicted_disease)
                self.speak("Here are some suggestions for you.")
                self.speak(action)

if __name__ == "__main__":
    try:
        MedBuddy()  # Instantiate the MedBuddy class
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        rospy.loginfo("MedBuddy node terminated.")  # Log termination message

