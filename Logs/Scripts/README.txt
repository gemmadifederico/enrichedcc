------------------------------------------------
SCENARIO 1
NIGHT ROUTINE
------------------------------------------------
The log contains three different activity: Bed, Bathroom and Get_up
Each case has a single trace.
Each case refers to a different day of recording, and it starts around 21 and ends around 9.
The process duration is in total from 7 to 8 hours.
Each log is composed by 1000 traces.
The format of the file is: caseid, activity, start timestamp, complete timestamp

##########################
NORMAL LOG logNormal.xes
##########################
During the night the person wakes up between zero and two times to go the bathroom. 
The duration of the bathroom activity is between 6 and 10 minutes.

##########################
FREQUENCY LOG logFreq.xes
##########################
During the night the person wakes up between four to eight times to go the bathroom. 
The duration of the bathroom activity is between 6 and 10 minutes.

##########################
DURATION LOG logDur.xes
##########################
During the night the person wakes up between zero and two times to go the bathroom. 
The duration of the bathroom activity is between 25 and 40 minutes.

------------------------------------------------
SCENARIO 2
EATING ROUTINE
------------------------------------------------
The log contains four activities: Eat, Leave, Enter, Relax
Each case has a single trace.
Each case refer to a different day of recording, and it starts around 11 and ends around 9.
Each log is composed by 1000 traces.
The format of the file is: caseid, activity, start timestamp, complete timestamp 

##########################
NORMAL LOG logNormal.xes
##########################
In a normal scenario, the person starts lunch between around 11:30 and 13:00. 
Then he leaves, to come back for dinner time, between 18 and 20.
After having dinner, he relaxes.
The duration of the eating activities are between 30 and 60 minutes.

##########################
DELAY LOG logDelay.xes
##########################
In a delayed scenario, the person has lunch between 14:00 and 15:00, and dinner on time.
Or, he has lunch on time and delayed dinner between 21:30 and 23. 
The duration of the eating activities are between 30 and 60 minutes.

##########################
ABSENCE LOG logAbsence.xes
##########################
In an absence scenario, the person skips lunch or dinner OR lunch and dinner.

------------------------------------------------
SCENARIO 3
MORNING ROUTINE
------------------------------------------------
The log contains five activities: Wake up, Have breakfast, Go bathroom, Get dressed, Go out
Each case has a single trace.
Each case refer to a different day of recording, and it starts between 7:00 and 8:00
Each log is composed by 1000 traces.
The format of the file is: caseid, activity, start timestamp, complete timestamp

##########################
NORMAL LOG logNormal.xes
##########################
In a normal scenario, the person starts with waking up, then he has breakfast. The duration of breakfast is between 20 and 45 minutes. Then the person goes to the bathroom (15 to 25 minutes), makes bed (5 to 10 mins) and get dressed (15 to 30 mins). At the end, the person goes out.

##########################
SHUFFLE LOG log.xes
##########################
In a shuffle scenario, the person does not follow the normal control flow, but mixes some activities.
The process can start with the get dressed activity, or execute it later. Then the have breakfast, go bathroom and get dressed are randomly shuffled. If the get dressed was not executed before, it is now added. At the end, the person goes out.