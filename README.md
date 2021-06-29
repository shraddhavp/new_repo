# Final year engineering project - Personality prediction system
👉Personality prediction is a system that is used to predict a person’s behavioral traits based on 
asking certain questions, by analyzing their Curriculum Vitae, analyzing their social media 
accounts, analyzing their handwriting etc. This project is carried out based on the asking the
user a certain amount of questions relating to the traits in the Big Five Model that is OCEAN 
(Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) and the user is has 
to upload his/her CV for the analysis of the field that the user is good in.

# Proposed system 👇

The proposed system allows the user to enterhis/her details such as name and age. Then the 
user has to upload his curriculum vitae (CV) according to the required format. The required 
format is attached as a downloadable link and the screenshot of it is also attached. If the 
format does not match then a message indicating the same is displayed. If the resume is 
uploaded in the correct format then a message indicating the successful submission of the details is displayed. Then the user has to take the test related to the questions based on the behavioral traits and then the results based on the answers given by the user the result is 
displayed in terms of percentage along with the field that the user is good in based on the 
information scanned from his resume and one word describing the user’s behavior. All these 
information is stored in a .txt file. The field that the user is good in is calculated based on an 
algorithm called Term Frequency (TF-IDF) algorithm taking user’s technical skills, projects, 
certification and work experience. The word describing the user’s behavior is calculated 
based on logistic regression algorithm.
The system also allows the admin to login with his credentials. Admin can see the candidate 
list and the links to the files that stores the candidate details that he can use for later purpose.


# Advantages of the system

The advantages of the existing system are:

* The system predicts personality of the person along with the prediction of the skill 
that the user is good at in the same system.
* The system gives more accuracy almost 85% in the case of logistic regression in 
predicting the personality of the person.
* It also has the word document of the resume template already embedded within it 
which can be downloaded and filled if the resume that user has is not in the specified 
format.

# Technologies and tools used
* Tensorflow
* Programming language -Python
* Keras
* Anaconda(Running python environment)
