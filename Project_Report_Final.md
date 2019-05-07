## ST449 Research Project Report

# Music Generation with Artificial Intelligence - *Creative sequence modelling using LSTM Recurrent Neural Networks*   



# 0. Project Overview 

The project examines how to generate musical melodies using artificial intelligence. The project goal was to examine how to effectively generate music melodies that capture the characteristics of a given melody sequence both in general and specific music contexts. This research goal was met by training three LSTM Recurrent Neural Networks on different dataset sizes, applying transfer learning to approach a domain specific model in iteration 2.2. Dataset choice, Model choice parameter options and customizations are informed by the requirements of the application context, application music practitioner literature, the authors own domain knowledge from 18 years as a music practitioner. The models entail customized implmentations of the Google AI Magenta models for melody generati (Google AI Magenta, 2019c).

*The project includes:* 

- Dataset building for general sequence modelling in musical domains
- Dataset building for sequence modelling in the specific application context. 
- Training a RNN LSTM for general sequence prediction
- Applying learning to further train the Model  towards a context domain specific application
- Empirical evaluations of trainig iterations in transfer learning.
- Design and application of context specific model evaluations mechanisms informed by application domain literature and practice. 

*Conceptual models laying the direction for further research include*

- Theorized and explored how the Magenta RL tuner could be adapted to use Deep Q Reinforcement Learning to tune this context specific model further using customized rewards from the application domain
- Theorized how this system could be trained to adapt to a performer in real time, using Online Reinforcement Learning and custom rewards aimed at musical collaboration.  

**Summary of Results**

The final numerical evaluations show that Iteratin 2.2 managed to achieve the best metric scores of the models, with Accuracy of 0.9012, Loss and Loss per step of 0.3132 and no_event_accuracy of 0.9901.

The general model Iteration 2.1 performed better on generating musical outputs than Iteration 1, validating the usefulness of training on a bigger dataset. Iteration 2.2 was designed to be context specific. Consequently, it performed better on context specific music generation than the general model 2.2, but worse on general music generation. This validates the model as a specialised context specific model, and illustrates the difference between general and context specific sequence generation models. 

| **Model** | **Step** |***Accuracy*** |***Loss***| ***Loss_per_step***| ***no_event_accuracy*** |
| --- |--- | --- | --- | --- | --- |
| Iteration 1| 20k |  0.7347|  0.8976 | 0.8976 | 0.9678 |
| Iteration 2.1 | 19.99k |0.7275 | 0.9203 |0.9203  | 0.9663 |
| Iteration 2.2 | 72.05k | 0.9012| 0.3132 | 0.3132 |0.9901|


# 1. Introduction and Research Question

## Project Goals & Context

### Application Goals

The application context of the music project is in the realm of generating melodies for real time music improvisations in a cinematic style, such as film and game music. The Goal is to design a music generator that is capable of generating melodies that are both musically expressive and have a certain coherency. The model will generate a one voice, aka. monophonic, melody line. 

### Research Goals / Question 

Thus, the project entails the implementation and evaluation of a neural network for a sequence modelling task in the context of music generation. This was based on the following research question:

*How to effectively generate music melodies that are able to capture and imitate the characteristics of a given melody sequence both in general and specific music style contexts* 

This was examined using the following models: 

-  *Iteration 1: A LSTM RNN model trained on a general musical dataset of 11 133 musical examples.* 

- *Iteration 2.1: A LSTM RNN trained on a larger general music dataset of 179 781 musical examples and including captured performances* 

- *Iteration 2.2: A LSTM RNN applying transfer learning by continuing training from 2.1 on a small selection of 530 context specific music examples.*

### Artificial Intelligence Research Context 

As with text and visual art generation, music generation falls within the space of computational creativity (Eck & Schmidhuber, 2002; Fernández & Vico, 2013; Franklin, 2009)   In addition to providing a number of usecases, it is also interesting in the sense that creativity is a fundamental aspect of human intelligence (Russel & Norvig, 2016). It is recognized that contemporary AI hold advantages over humans in analytical tasks such as identifying relationships and caputuring complex dynamics of large datasets. However, creativity is argued to be one of the areas in which humans still hold significant (Parry et. al, 2016 p.580; Jarrahi, 2018 p.582). Exploration of computational creativity is therefore an important step on the path to general or strong Artificial Intelligence (Russel & Norvig, 2016 p.16; Jaques et. al.,2017). Music generation with Artificial Intelligence is situated in a subfield of computational creativity called "Algorithmic Composition", which since the 1950´s have applied a range of AI techniques. In recent years, approaches built on artificial neural networks have gained significant importance and advances in the field could see AI systems enhance the creative output of human composers similar to how expert systems aid many modern day professionals (Fernández & Vico, 2013).  

# 2.0 Choice and Description of Data 

## Musical Data & Patterns

**Midi format** 
Similar to earlier research (Franklin, 2019), this project focuses on digital music at the pitch and duration level. This data is represented in the standard MIDI file format, which among other features allow 128 different pitches to be represented with along with their duration and velocity (The MIDI Association, 2019). 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/bach_notes.png) 

**Figure 2.1: Musical Notation of Melody, Bach Cello Suite 1**

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/bach_midi.png) 

**Figure 2.2: Midi Notation of Melody, Bach Cello Suite 1**

Figure 1 and 2 show the start of Bach´s Cello Suite nr.1 as both standard musical notation and midi notation. As we can see, midi notation is essentially a graph with pitch on the y-axis and time on the x axis. Music can therefore represented as some function  <img src="http://latex.codecogs.com/svg.latex?f" border="4"/> which caputure a value state <img src="http://latex.codecogs.com/svg.latex?s" border="4"/> at time <img src="http://latex.codecogs.com/svg.latex?t" border="4"/>. 

### Musical Expressions

The data used for the model consists of musical material from the western musical traditions.  Although developing through history, the fundamental building blocks of this music have persisted from Bach (18th Century) to contemporary popular music (Persichetti, 1961; Pease, 2004). A selection of these are: 

**Musical Note:** A pitch sounding over an arbitrary lenght of time (Persichetti, 1961; Pease, 2004). 

**Monophonic melody** Comes from "one voice" and refers to any instance where only a single note, aka. a single sequence value is present at any one time (Persichetti, 1961). This research project focuses on monohponic melody generation. 

**Melody**: Common use of the term refers to an arbitrary length sequence of note values. More precise definitions are of a more philosophical nature, and of little practical importance (Persichetti, 1961; Pease, 2004). 

**Harmony**: Two or more notes played at the same time, aka. two or more values present. Notes played in sequence can also be perceived as creating harmonic material, such as when all the notes in a chord are played after each other (Pershichetti, 1961).

**Intervals**: The value difference between two notes, in harmony or in sequence. Absolute intervals refer to absolute value differences and relative intervals refer to relative value differences (e.g. Persichetti, 1961: Pease & Pulling, 2001).

**Octave:** A note value difference of 12 half notes, or 8 notes in most common scales. These have the same letter names (Persichetti, 1961; Pease & Pulling, 2001). 

**Themes/Motifs** :  A local structure consisting of a shorter sequence of note values (Eck et. al., 2002;Miller, 1996;Pease, 2004). If a note is a single character, a motif is a word. If in doubt, listen to the first four notes of Beethoven´s 5th Symphony. 

**Phrase:** A longer local music structure often built up by several motifs (Eck & Smidhuber,2002; Pease, 2004). The musical equivalent of a sentence. 

**Thematic & Melodic Development** : Variating of a local sequence structure to form a developing global structure. Sometimes, 
a local pattern of relative value differences are repeated in with varying absolute note values (Miller, 1996; Pease, 2004). If in doubt, listen to the first thirty seconds of Beethoven´s 5th Symphony. 

**Scale:**  An arbitraty collection of unique sequence values within an octave; within 12 half note values. Most common scales in Western music and styles derived from this have 7 unique note values(Persichetti, 1961; Miller, 1996). 

**Modal:** Often used to refer to musical expressions using other scales than conventional major and minor (Persichetti, 1961; Pease & Pulling, 2001; Pease, T. 2004). 

**Chords**: A Local music structure. Three or more notes perceived to be sounding at the same time either by sounding at the same time step, or by appearing in close proximity to each other in a sequence. The latter applies in monophonic melodies (e.g. Persichetti, 1961; Pease & Pulling, 2001)

**Harmonic Development/ Chord progression** :  meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development).   (e.g. Persichetti, 1961; Pease & Pulling, 2001). 

**Rhythm**: The relationship between the length of different pitch on instances in a sequence, and between on and off instances (Persichetti, 1961; Miller, 1996;2000; Pease, 2004; Pease & Pulling, 2001). 

## Datasets

The data consisted of midi files and was selected with the overall application goal in mind, including both general music data ans style specific. Midi files were inspected using the musical notation software Sibelius, and output melodies were inspected with the Digital Audio Workstation Ableton Live. 


### General music data

The General music set consisted of two parts. 
- *The Lakh Midi Dataset dataset*, which was collected for by Raffel(2016) for Phd work on Audio-to-midi processing.  It is made up of 178 579 midi files scraped from publicly available web sources (Raffel, 2019). 
- *The Maestro Dataset* , contructed by Google AI Magenta. It contains 1 202 midi files capturing performances of classical piano pieces from world class performers  (Google AI, Magenta 2019b) 

### Style Specific music data

The style specific dataset was selected as specific influences for the context specific music application of the AI. It consists of 530 files combining both polyphonic and monophonic instances, covering movie, video game and TV themes, as well as a selection of Classical and Jazz pieces. 

**Movie themes:**  A subset of movie themes with themes of a symphonic nature were emphasized (midiworld, 2019a) 

**Video Game themes:** A subset of Video Game theme, with themes of a symphonic nature were emphasized (midiworld, 2019b) 

**TV Themes:** A subset of TV themes (midiworld, 2019c)

**Modal Classical:** Classical music from selected composers were also included as part of the style specific dataset. This was composers who were in the expressionist, impressionist and nasjonal romantic with inspiration of Scandinavian folk music  paradigms as this was the moderns sounding style that would suit the model context (Persichetti, 1961) 

**Jazz Data:** The purpose of including the jazz solos was for the model to learn jazz phrasing and performance, with specific focus on improvsational melody construction. Most jazz material is written down only in sparse form, with practitioners given ample freedom to improvise and interpret hte material. Most practitioners therefore learn by imitating and transcribign recordings rather than playing after sheet music .  This tanscription of individual performances is a slow and incredibly time consuming task, which practitioners spend years perfectin. Books are released with transcriptions of individual performances, but these are very rarely in midi format (Giddins & Devaux, 2009). For capturing sufficient data sets to train AI models it is therefore nessecary to source transcribed midi jazz from the web, and the material for the jazz part of the dataset was sources from The Jazzomat Researhc Project (2019) and Acroche2 Studio (2019). A selection of 64 jazz solos and 58 ensemble pieces and where included in the the dataset based on the artists emphasis on modal melodies and modern harmonic colours. These were solos by Miles Davis, John Coltrane, Herbie Hancock, Wayne ShorterKenny Wheeler and Weather Report (Giddins & Devaux, 2009).  

# 3. Solution Concepts 

## Models

The following models were created: 

- **Iteration 1:** A Note based LSTM RNN for General Music Generation trained on a subset of the General Music Dataset, only including musical composition.
- **Iteration 2.1:** A Note Based LSTM RNN for General Music Generaton trained on the entire General Music Daset, including both musical compositions and captured performances. 
- **Iteration 2.2:** A Note Based LSTM RNN which applied transfer learning to adapt Model 2.1 to the application context domain by training it on the style specific dataset.

### LSTM Recurrent Neural Networks for sequence generation 

The context application and research question entailed how to generate a monophonic, aka. single voice musical melody capturing and imitating the characteristics of a given melody in general and specific music contexts. In AI terms this entails generating a sequence of states which capture and immitate the characteritics of a given sequence of states. Music generation can considered a dynamic system, aka. a feedback proces, since the current state is dependent on a history of past states (Franklin, 2019).

Applying the classical form of a dynamical system (Goodfellow et. al., 2016), a music system can therefore be written as 

<img src="http://latex.codecogs.com/svg.latex?s^t=f(s^{t-1}; \theta)" border="4"/>

with <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/> being the current state of the music, <img src="http://latex.codecogs.com/svg.latex?s^{t-1}" border="0"/> being the state at the previous time step and  <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/> capturing the system parameters from the whole sequence of states. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/dynamical_music.png)
**Figure 3.1 : Melody as dynamical system, musical notation** 
![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/dynamical_music_midi.png)
**Figure 3.2 : Melody as dynamical system, midi data** 

Figure 3 and 4 illustrate the monophonic melody case. <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/> correspond to the current note while  <img src="http://latex.codecogs.com/svg.latex?s^{t-1}" border="0"/> corresponds to the preceeding note.  <img src="http://latex.codecogs.com/svg.latex? \theta" border="0"/> is applied to all time steps , incorporates information about the whole sequence (blue). In the above example, this could be used to infer the note at  <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/>  by including information that the starting note repeats after 8 time steps(red line). It could also capture that the melody consists of two alternating unique patters of 4 values (purple and yellow lines), with the pattern beginning at <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/>  has the same start note value as the melody starting note.  


This makes music generation a sequence modelling task that involves generating a sequence of outputs given a sequence history of inputs.  Sequence2Sequence Recurrent Neural networks are therefore good model candidates (Géron, 2017 p.407; Goodfellow, Bengio & Courville, 2016). While Feed Forward and Convolutional Neural Networks generate a fixed size output from a fixed size input, Recurrent Neural Networks are able to generate output sequences of arbitrary length from input sequences of arbitrary length (Géron, 2017 p.407). This also supports the use of an RNN model for creating a music generation model responsive to improvisation, as improvised musical phrases often vary in lenght (Giddins & Deveaux, 2009). 

**LSTM for Music Generation** 

A challenge in Recurrent Neural Networks is that the gradients tend to either vanish or explode when they are propagated over many stages. This property makes it hard for regular RNN models to learn long term dependencies (e.g. Goodfellow et. al., 2016 p. 390). An essential feature of most musical improvisations and compositions is the repetition, developments and self reference of musical patterns over a high number of sequence steps (Giddins & Deveaux, 2009; Persichetti, 1961; Pease, 2004). In order to perform well, musical generation models must therefore address this challenge. Indeed, research has found that models composed of standard Recurrent Neural Nets fail to generate music with global coherence (Eck et. al., 2002). 

The Long Short-Term Memory Recurrent Neural Network uses gated self-loops to solve the challenge of long term dependencies in RNN´s. The self loops allow gradients to flow through many time steps and the gates are controlled by state context, allowing information to flow through many time steps  (Géron, 2017p.407-409; Goodfellow et. al., 2016 p.397-99; Hochreiter & Schmidhuber, 1997). In the research literature, LSTM Recurrent Networks have been used to generate monophonic melodies (Fernández & Vico, 2013), including Improvisational patterns (Eck & Schmidhuber, 2002; Franklin, 2019). Considering the end goal of the current application is also in the realm of improvisation, with a subset of the data coming from Jazz, the latter two papers make the LSTM RNN model especially relevant.  Similar to character level RNN´s (Graves, 2013), the AI music generator model trained in iteration 1, 2.1 and 2.2 uses a LSTM RNN predicting the next data point from a series of input values (Abolafia, 2016; Jaques et. al.,2017; Mikolov et. al.,2010) In the RNN LSTM, the token used for prediction is known as the "primer" for the model (Abolafia, 2016; Google AI Magenta, 2019c). The model is built upon the Melody RNN model by Google Magenta, which implements the basic tensorflow LSTM cell ``` tf.contrib.rnn.BasicLSTMCell()```(Abolafia, 2016; Google AI Magenta, 2019c). 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/LSTM_cell.png)

**Figure 3.3: LSTM RNN Cell (Géron, 2019)**


Figure 5 illustrates the workings of the LSTM Cell. The cell uses two distinct states, the short term state <img src="http://latex.codecogs.com/svg.latex?h" border="0"/> and the long term state 
<img src="http://latex.codecogs.com/svg.latex?c" border="0"/>. The long term state traverses the cell first through a forget gate , where some memories are forgotten, then through an input gate which adds some new memories. In music, these gates could for instance trigger the long term state to forget a certain rhytm pattern or probablitity of some note values when the rhytm and scale of the music changes during a new section. After tanh filtering, this long term state is also passed along as the next short term state <img src="http://latex.codecogs.com/svg.latex?h_{(t)}" border="0"/>. Both the short term state <img src="http://latex.codecogs.com/svg.latex?h_{(t-1)}" border="0"/> and the input vector <img src="http://latex.codecogs.com/svg.latex?x_{(1)}" border="0"/> act on the four fully connected layers the main layer <img src="http://latex.codecogs.com/svg.latex?g_{(t)}" border="0"/>,forget gate controller  <img src="http://latex.codecogs.com/svg.latex?f_{(t)}" border="0"/>,the input gate controller  <img src="http://latex.codecogs.com/svg.latex?i_{(t)}" border="0"/> and the output gate controller  <img src="http://latex.codecogs.com/svg.latex?o_{(t)}" border="0"/>. The Main layer  <img src="http://latex.codecogs.com/svg.latex?g_(t)" border="0"/> analyzes the current inputs  <img src="http://latex.codecogs.com/svg.latex?x_{(t)}" border="0"/> and short term state  <img src="http://latex.codecogs.com/svg.latex?h_{(t-1)}" border="0"/> and passes this partially to hte long term state. The forget gate controller  <img src="http://latex.codecogs.com/svg.latex?f_{(t)}" border="0"/> controls the forget gate, while the input gate controller  <img src="http://latex.codecogs.com/svg.latex?i_{(t)}" border="0"/> controls what parts of the main layer output from  <img src="http://latex.codecogs.com/svg.latex?g_{(t)}" border="0"/> will be added to the long term state. The output gate  <img src="http://latex.codecogs.com/svg.latex?g_{(t)}" border="0"/> controls what parts of  <img src="http://latex.codecogs.com/svg.latex?c_{(t-1)}" border="0"/> gets output to  <img src="http://latex.codecogs.com/svg.latex?y_{(t)}" border="0"/> and  <img src="http://latex.codecogs.com/svg.latex?h_{(t)}" border="0"/>. Through this process, LSTM learns to recognize and store an important input such as the starting or root note in the long term state, and read it when needed (Géron, 2017). 


### Other Model Specifications 

The remaning model specifications include: 

**Training Input**

A sequence of pitches over 128 possible midi values with a given lenght. For a given training melody sequence at step <img src="http://latex.codecogs.com/svg.latex?t" border="0"/>, the input sequence value at time step <img src="http://latex.codecogs.com/svg.latex?t" border="0"/> corresponds to sequence entry <img src="http://latex.codecogs.com/svg.latex?n" border="0"/>.  

**Training Labels** 

A sequence of pitches over 128 possible midi values with a given lenght, with each sequence value corresponding to the next step sequence value of the input sequence. For a given training melody sequence , the input sequence value at time step <img src="http://latex.codecogs.com/svg.latex?t" border="0"/>  thus corresponds to sequence entry <img src="http://latex.codecogs.com/svg.latex?n+1" border="0"/>.   

**Generation input: Primer**

An arbirary length sequence of pitches over 128 possible midi values 

**Generation Outputs**

An arbitrary length sequence of pitches over 128 possible midi values, capturing primer sequence pattern characteristics, i.e musical attributes. The precise length is specified in the model parameters.  


### General Music and specific models with Transfer Learning

The music model was be first trained as a general model, then adapted to the specific application using transfer learning.  
This approach entails using a model trained on a general dataset as starting point for further training of a model on a specific dataset. This is in order to make the second model more general, and it is build on the assumption that many of the factors explaning variations in the first dataset hold in the second.  (Goodfellow et. al.,2016). In Language this could entail building a general model for the language in question, then specialising the model to a certain application field using application specific data (Conneau, Kiela, Douwe, Barrault, Loic, Schwenk, & Bordes, 2016 ;Zoph,Yuret,  May, & Knight, 2016). In music, this training process could be utilised by first training the model on a general music data from which the model could learn general musical rules and conventions. The model could then be further trained on a data from specific style of music in order to specialise it to this field. 

Both general music data and the context specific data are governed by the basic musical building blocks described in the Musical Data setion (Persichetti, 1961; Pease & Pulling, 2001). These musical building blocks are analogous to the low level visual features captured in CNN visual systems, where transfer learning is frequently applied to adapt models trained on one visual category to another (Goodfellow et. al., 2016). 

# 4. Model Implementation 

### Data Preprocessing using Magenta´s Command Line API 

The data was preprocessed using the Magenta Command Line API. This was done in two steps, first data had to be converted from midi files to a tfrecord file of notesequences. Notesequences are protocol buffer files enabling efficient data handling during training (Google AI MAgenta, 2019c). This process was done for all data subsets: 

```
INPUT_DIRECTORY=/Volumes/Christians_Drive/ai_music/context_datasets/jazz/
SEQUENCES_TFRECORD=/Volumes/Christians_Drive/ai_music/context_datasets/jazz/notesequences.tfrecord
convert_dir_to_note_sequences \
--input_dir=$INPUT_DIRECTORY \
--output_file=$SEQUENCES_TFRECORD \
--recursive
```

These notesequence protocol buffers then had to be structured into SequenceExamples, a file that contains a sequence of inputs and a sequence of labels that represents a melody. This command also splits the set into a training and test dataset on the `eval_ratio` parameter. The `config`parameter was specified to the type of model that was going to be trained. The Magenta LSTM RNN implementation includes several options. The 'mono_rnn' configuration was selected for the model as this one allows notes to take all 128 midi values, giving a wider range of musical output. 

Command line code Example: 

```
melody_rnn_create_dataset \
--config='mono_rnn' \
--input=/Volumes/Christians_Drive/ai_music/context_datasets/jazz/notesequences.tfrecord \
--output_dir=/Volumes/Christians_Drive/ai_music/context_datasets/jazz/sequence_examples \
--eval_ratio=0.10

```

### Model Implementation 

Below are some additional comments and considerations for the implementation. The full implementatin code can be found at: https://github.com/lse-st449/st449-projects-casvalesen/blob/master/Music_Generation_with_AI_Code_Implementation_Notebook_v_0_4.ipynb 


**Gdrive integration**  

The preprocessing was done on a local hardrive. The data was then uploaded to gdrive, with was mounted to Colab using the standard Colab mount function: 
```
#Mount drive to Google Colab 
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)   #/content/gdrive/My Drive/AI_music/ , force_remount=True
root_path = 'gdrive/My Drive/AI_music/'

```
This ensured files could be called directly from the colab and all processing could be cloud based. 

**Model Building and Training** 

The Magenta distribution (Google Magenta AI, 2019c) supports commmand line training of RNN based models. However, to build a customized model using the library it was necessary to thoroughly comprehend and apply sections of the Magenta distribution source code. The Melody RNN Model  distribution consists of eight interlinked python modules: 

```melody_rnn_config_flags.py``` Provides a class for model configuration, utility functions and defaults. Most importantly, it implements ```tf.app.flags.FLAGS```to set hyperparameters for the model. In order to customize trained model, these FLAGS had to be managed and configured manually in the custom implementation. When called directly, a bug create a frequent warning for misssing flag 'f', thus a dummy flag was defined in the implementation to deal with this error. 

```melody_rnn_generate.py``` This module was called to generate output sequences from the trained model. It needed to be configured with a separate instance of ```FLAGS```. 

```melody_rnn_model.py``` This is the module for the main class for the model itself.

```melody_rnn_train.py``` This module specifies specific training operations for the model. 

```melody_rnn_sequence_generator.py``` This model generates output sequences from the model once trained. 

```melody_rnn_pipeline.py``` This model allows for implementation for custom data pipelines. As data was preprocessed using the command line API, the module was not used. 

```melody_rnn_create_dataset.py``` This was the module called during preprocessing with the command line API. Not necessary to deal with further in implementation. 

```melody_rnn_create_dataset_test.py``` Provides tests for the above, largely irrelevant for model customization. 
 
The Melody RNN Model distribution also calls modules shared accross models, namely: 

```from magenta.models.shared``` : 

``` events_rnn_graph```  Defines the low level tensorflow graph for the model, includign softmax cross-entropy loss and Adam Optimiser.  The function ```get_build_graph_fn()``` is called directly in the project implementation. 

```events_rnn_train``` This module runs contains the ```run_training```function that runs the training of the LSTM RNN graph. In the custom implementation this function was called directly with specified arguments. 

Several of the subfunctions take ```unused_argv```as inputs. After a visit to stackoverflow, it was apparent this is internal code practice at Google, and the variable can be set to   ``` unused_argv= ' '``` to avoid any issues. 

### Custom_functions Class 

A Custom_functions class was defined to implement functions necessary to customize the Magenta Melody RNN Model. It includes:

- **inspect_configs()** to inspect current configurations

- **evaluate()** wrapping the .run_eval to automatically reuse hparams from training.

- **generate_test_melodies()** generating test melodies for the current model iteration

- **show_generated_outputs()** to display generated outputs as notesequences in the notebook

- ***Custom function: generate_test_melodies()*** 

An ``` generate_test_melodies(checkpoint_file,run_dir,output_dir) ``` was defined to generate all the outputs necessary for model evaluation (for specific evaluation metrics see below). It takes a the following arguments: A ```checkpoint_file``` which is the checkpoint for a specific model iteration to be used and a ```output_dir``` which is the directory to the mounted GDrive where the output melodies should be saved. 

In the distribution function ``` melody_rnn_generate.main(unused_argv) ```, ``` FLAGS.run_dir``` contains specifies a directory from which the model will load the latest checkpoint for use in sequence generation. By default, ```run_dir```is set to ``` None ``` to avoid conflict with the ```checkpoint_file``` flag. However, if one should wish to load the latest checkpoint in a directory instead of a specific checkpoint file, one can set ```checkpoint_file``` to ```None``` and specify a ```run_dir``` instead. 

The function first specifies ```FLAGS```that are shared among the differnt primers, then specifies ```FLAGS ```for each of the evaluation primer melodies. The primer name is added to the ```output_dir``` to generate different subfolders for the data and a message is printed so show runtime success at each step. ```melody_rnn_generate``` takes either a midi file directory in ```primer_midi``` or a  notesequence in ```primer_melody``` as inputs. As the first four output examples where defined using notesequences to ```primer_melody```, ```primer_midi``` was initialised to ```None```. This was reversed and overwritten when the model takes a bach midi file as input. 

Full function below: 

```
#Function to generate Evaluation test melodies from trained models
def generate_test_melodies(checkpoint_file,output_dir, run_dir=None):

   
  #Shared flags 
  FLAGS.config='mono_rnn'
  FLAGS.checkpoint_file=checkpoint_file
  FLAGS.melody_encoder_decoder=None
  FLAGS.run_dir=run_dir 
  FLAGS.num_outputs=5 
  FLAGS.num_steps=128 
  FLAGS.hparams="batch_size=64,rnn_layer_sizes=[128,128]"
  #Primer midi set to None, get overridden in Bach cello step 
  FLAGS.primer_midi=None 
  unused_argv='_' 

  #Generate from one note  
  FLAGS.primer_melody="[60]"  
  FLAGS.output_dir=output_dir+'onenote/'
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from One note primer")
  
  #Generate from half note  
  FLAGS.primer_melody=None
  FLAGS.primer_midi='/content/gdrive/My Drive/AI_music/primer/half_note_primer.mid'
  FLAGS.output_dir=output_dir+'half_note/'
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from half note primer")
   
  #Generate from simple major melody 
  
  FLAGS.output_dir=output_dir+'simple_major/'
  FLAGS.primer_melody="[60, -2, 62, -2, 64, -2, 59, -2,60,-2]" 
  
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from Simple major Melody primer")
  
  #Generate from simple minor melody
  FLAGS.output_dir=output_dir+'simple_minor/'
  FLAGS.primer_melody="[60, -2, 62, -2, 63, -2, 58, -2,60,-2]"
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from Simple Minor Melody primer")
  
  #Generate from arpeggiated chord sequence

  FLAGS.output_dir=output_dir+'arpeggiated'  
  FLAGS.primer_melody="[62, -2, 65, -2, 69, -2, 72, -2,71,-2,67,-2,65,-2,62,-2,60]"
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from Arpeggiated Chord sequence primer")

  #Generate from bach cello 
  FLAGS.output_dir=output_dir+'bach/'
  FLAGS.primer_midi='/content/gdrive/My Drive/AI_music/bach_gen/cs1-1pre_4bars.mid'
  FLAGS.primer_melody=None
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from Bach Cello primer")
  
  #Generate from context specific modal motive
  FLAGS.output_dir=output_dir+'modal/'
  FLAGS.primer_midi='/content/gdrive/My Drive/AI_music/primer/modal.mid'
  FLAGS.primer_melody=None
  melody_rnn_generate.main(unused_argv)
  print("Melodies generated from Modal context primer")
  
  
  return None
```


# 5. Evaluation 

## Numerical Evalation mechanisms 

The data was inspected using tensorboard. Standard neural network performance metrics were used to evaluate the models: 

**Accuracy:** The ratio of correctly predicted values  (Géron, 2017). 

**loss_per_step:** The amount of loss for each training step. The melody rnn model defines this as the Mean Softmax cross-entropy loss, implementing the loss function ```tf.nn.sparse_softmax_cross_entropy_with_logits```. This loss function trains model to estimate high probabilities for the target class by penalizing low target class probabilities (Géron, 2017 p.143). This loss function suitable for the task of modelling musical sequences as it captures whether the right note was predicted rather the absolute value difference of the prediction. In a musical context, absolute value difference is not a meaningful measure, as being half step wrong can often be sound musically worse than being three or four half steps wrong (Persichetti, 1961).  
 

**no_event_accuracy:** The ```MELODY_NO_EVENT``` indicator is implemented in the Magenta library to mark when a sustained state is held over several values, which means the current status of ```NOTE_OFF``` or ```NOTE_ON``` is kept. In the library source code, ```No_event_accuracy``` is defined as the sum of the products of correct_predicitions and no_events, divided by the sum of no events. 

```
   no_event_accuracy = (
            tf.reduce_sum(correct_predictions * no_event_positions) /
            tf.reduce_sum(no_event_positions))
```
(Google AI MAgenta, 2019c)

This is therefore a music specific numerical evaluation which gives a measure of how accurate the value durations, aka. rhytms, sequences are compared compared to the target sequences. 


### Domain Specific Evaluation 

As a creative RNN is in its nature not trying to predict an accurate value such as a time series or classification, output
evaluation of creative RNN is in its nature somewhat subjective.  Although accuracy rate might give an objective measure of how well the model predicts a melody note, it is not necessarily informative on how the model performs in context specific terms, namely in how it manages to be musically expressive and use established musical conventions.  To overcome this challenge, researchers have used human evaluators both to train and to to judge the outputs of Artificial intelligence music systems both through crowdsourcing to get domain specific evaluation statistics (Jaques et. al.,2016) and through individual evaluators (Fernández & Vico, 2013).  


**Evaluation Criteria**


A known failure mode of single step prediction RNN´s in sequence generation is the continuous repetition of the same token(Jaques et. al., 2017). The results of the LSTM RNN  were therefore evaluated on whether this failure mode was occuring. 

Another common failure mode of single step RNNs is the difficulty of ensuring coherent global structure. In music specifically, this global structure is built up by musical phrases (Jaques et. al., 2017). An evaluation criteria for the generated sequences is therefore to what extent they have a coherent global structure.

In a musical context this can refer to the presence of the following characteristics (Persichetti, 1961; Pease, 2004): 
- *Motives & themes*
- *Motific & thematic development*
- *Melodic development*
- *Harmonic development*
- *Consistent scale/key* 

For the purely generative models, evaluation is based on how well the RNN LSTM model has been able to learn general musical parameters. Note sequences were generated based on several different primers, which aimed at 

*Primer: Starting note*

This examines what the AI generates freely when it does not try to match the musical characteristics of a specific melody. 

*Primer: Simple Major melody* 

This examines what the AI generates when it is fed a simple major melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the major scale
- imitate stepwise melody movement 
- use a leading note half step below the root note to lead back to the root note in melodic movements. 
- capture the steady melody rhytm with (...) as main sequence step length. 

*Primer: Simple Minor Melody.* 

This test is equivalent to the major melody, just in minor. It examines what the AI generates when it is fed a simple minor melody, with two ascending steps and one step leading back ot the melody. It tests whether the melody is able:
- to generate melodies in the minor scale
- imitate stepwise melody movement 
- use a leading note whole step below the root note to lead back to the root note in melodic movements. 

*Primer: Arpeggiated chord progression* 

This tests whether the model is able to capture arpeggiated chords, aka. individual chord notes in sequence, and chord progressions. 

*Primer: Bach Cello Suite Excerpt*

Thi tests whether the model is able to campture the melody style represented in the start of Bach´s cello suite. The style is characterised by  16th notes, which are short sequence step lengths, a combination of stepwise motion (delta of one or two half steps) and larger interval leaps (sequence step deltas between [6-9] ]. 


**Output Naming Convention:** it_(iteration_nr)_(primer)_nr_(outputnr)_(training_ep if applicable)

### Model Iteration 1: Mono_rnn based on Magenta´s RNN LSTM Model 

Model Iteration was trained on the first 11 136 midi files from the Lakh dataset, using 20k training steps. 

**Model Outputs & Evaluation**

| **Model** | **Step** |***Accuracy*** |***loss***| ***loss_per_step***| ***no_event_accuracy*** |
| --- |--- | --- | --- | --- | --- |
| Iteration 1| 20k |  0.7347|  0.8976 | 0.8976 | 0.9678 |

Accuracy (Figure 5.1), Loss (Figure 5.2) and Loss Per step  (5.3) improved during training, with value graphs showing the typical decelerating improvement as is typical in Neural Network training (Géron, 2017). Of all the measures, we can observe that the no_event_accuracy (Figure 5.4) shows the highest result variance. This graph also first sharply increases, then descreases before reaching a plateau. However, it still has a final value of 0.9678. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_accuracy.png)


***Figure 5.1: Iteration 1 Accuracy***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_loss.png)

***Figure 5.2: Iteration 1 Loss ***


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_loss_per_step.png)

***Figure 5.3: Iteration 1 Loss_per_step***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_no_event_accuracy.png)

***Figure 5.4: Iteration 1 no_event_accuracy***

**Output Evaluation** 

*Primer: One note* 

For Iteration 1 the outputs generated from the one note primer largely follow one scale at a time and show occasional local patterns which could be start of motifs. However, they lack most forms of both local and global structure.  

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_onenote_example.png)

***MIDI 1: It 1 Onenote example***

*Primer: Simple major Melody*  

The model performs better on the major melody primer. It is able to capture the major key, the major rhytmic building blocks, and mimic some of the local melody patterns. However, it does suffer from surplus repetition, and occasionally goes out of key. It also no global structure patterns. 


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_major_example.png)

***MIDI 2: It 1 Major example***

*Primer: Simple minor Melody*

 For the simple minor primer, ITeration 1 generates more sporadic outputs. The most of the melody examples capture some of the key of the melody, but sometimes also includes chromatic notes aka. values not in the scale. It also introduces a greater variety of rhytmcs than for the simple major primer, suggesting either that the minor key material in the dataset is more varied rhytmically, or that the model interprets a variation in key as a variation in general, throwing in rhythmic variations as well. 
 
![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_minor_example.png)

***MIDI 3: It 1 Minor example***

*Primer: Arpeggiated chord progression*

The melodies generated from the arpeggiated primer in Iteration 1 manage to capture the major key of the primer sequence, however, they do are not able to capture the broken chords and generate further chord structures. Instead, the output includes a somewhat sporadic combination a combination of small and large interval jumps, lacking global structure. It is able to capture the main rhythmic building block.  

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_arp_example.png)

***MIDI 4: It 1 Arpeggiated example***

*Primer: Bach Cello Suite 4 bars 


The outputs for the Bach primer were able to capture the rhythm lenght of the primer melody as well as the major key. It also included several large interval jumps, suggesting the model picked up on some of the interval characteristics of the primer. However, several of the examples suffered from excessive repetition of notes. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_1_bach_example.png)

***MIDI 5: It 1 Bach example***


### Iteration 2: Mono_rnn based on Magenta´s RNN LSTM Model, complete general dataset (2.1) and style specific dataset (2.2)

**2.1: Complete General Dataset

**Model Outputs and Evaluation***

| **Model** | **Step** |***Accuracy*** |***Loss***| ***Loss_per_step***| ***no_event_accuracy*** |
| --- |--- | --- | --- | --- | --- |
| Iteration 2.1 | 19.99k |0.7275 | 0.9203 |0.9203  | 0.9663 |


For Iteration 2.1 both Accuracy (Figure 5.5), Loss (Figure 5.5) and loss per step (Figure 5.6) improve during the course of the training. Compared to Iteration final values are 0.0072 lower for accuracy, 0.0015 lower for no_event_accuracy and 0.0227 higher for the losses. This could be attributed to iteration 2.1 being trained on a larger dataset, thus taking slightly longer to converge. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_accuracy.png)

***Figure 5.5: Iteration 2.1 Accuracy***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_loss.png)

***Figure 5.6: Iteration 2.1 Loss**

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_loss_per_step.png)

***Figure 5.7: Iteration 2.1 Loss_per_step***


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_no_event_accuracy.png)

***Figure 5.8: Iteration 2.1  no_event_accuracy***


For the second iteration, a sixth primer melody was added. This was a simple phrase which would be similar to what would be used by the musical instrument player in the improvisation application context. 

**Output Evaluation** 

*Primer: One note* 

The melodies from the one note primer largely follow the same key, and have some local motif structures. They still lack global structure. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_onenote_example.png)
***MIDI 6: It 2.1 onenote example***

*Primer: Simple major Melody* 

The melodies from the simple major primer still capture both major rhythmic building blocks and the major key. For iteration 2.1 only one example melody included chromatic notes not in the key/scale, suggesting that the model was able to better learn scales and keys from the larger dataset. Also captured the stepwise intervals of the primer melody by generating many stepwise movement patters. One of the examples(MIDI 7) also included thematic development where the model almost directly imitated the primer pattern of first first playing three ascending notes, then approaching a target note from below (bars 5-7). 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_major_example.png)
***MIDI 7: It 2.1 major example***

*Primer: Simple minor Melody*

The minor examples also captured the main rhythmic blocks and the minor key of the melody. As in iteration 1, some of the output melodies also included more chromatic material. However, when this was used it was either a note consistently returned to, suggesting the model interpreted the minor key as a different scale, or as a chromatic note between two scale note. The latter is interesting as this is a melody movement frequently used in music with minor harmonic material. This suggests the larger dataset allowed the model to learn to imitate more characteristic melody movement. As with the simple major, the model also imitates many stepwise melody movements. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_minor_example.png)
***MIDI 8: It 2.1 minor example***

*Primer: Arpeggiated chord progression*

The model was still strugglign with the arpeggiated primer, and most of the examples were not coherent or captured broken chords. There was some improvement, however, as one of the examples (MIDI 9) was able to imitate the melody shape of the primer, and repeat versions this local stucture four times to form an overall global structure. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_arp_example.png)
***MIDI 9: It 2.1 Arpeggiated example***

*Primer: Bach Cello Suite 4 bars 

The outputs from the bach primer were still able to capture the key and the key of the primer sequence. It was also able to imitate the large interval jumps and the local motive structure of alternating between two higher notes and a lower one. Some of the examples also included several broken chords, suggesting the model learned to pick up on the broken chords used in the bach example. There were also fewer repeated notes in this iteration.  


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_bach_example.png)
***MIDI 10: It 2.1 Bach example***

*Modal* 

The modal primer was a new addition in this iteration. 
It was a modal melody in the scale of dorian minor. The model outputs were able to generate the major rhythmic building block of the primer, namely notes strctuning over two or more beats. The model also largely picked up on the scale of the primer, generating sequences in a minor scale.  However, the model occasionally also included chromatic non-scale notes.


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.1_modal_example.png)
***MIDI 11: It 2.1 Modal example***


### Iteration 2.2: LSTM RNN Context Specific Model** 


**Model Outputs & Evaluation**

| **Model** | **Step** |***Accuracy*** |***Loss***| ***Loss_per_step***| ***no_event_accuracy*** |
| --- |--- | --- | --- | --- | --- |
| Iteration 2.2 | 72.05k | 0.9012| 0.3132 | 0.3132 |0.9901|


As specified above, iteration 2.2 was trained on the style specific dataset of 530 midi files, continuing training from Iterations 2.1. A memory error during training caused the metrics data between step 19.99k and 25.37k to be lost. However, we can still infer the relevant key insights from the saved data. The final values of, Accuracy (0.9012), Loss (0.3132), Loss_per_step (0.3132) and no_event_accuracy (0.9901) were significantly better at the end of training than than in iteration 2.1. Comparing the training graphs of 2.1 and 2.2 we can observe a step change in these, with model 2.1 training trajectories flattening towards the end of training, and model 2.2 metrics flattening on a relative step change improvement from these (Figure 5.10, Figure 5.12, Figure 5.14). This is likely caused by the fact that iteration 2.2 uses a smaller sample, and that the model was already trained on general music characteristics from iteration 2.1, making it easier to predict the melodies in the context specific dataset. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_accuracy.png)

***Figure 5.9: Iteration 2.2 Accuracy***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2_accuracy_comp.png)


***Figure 5.10: Iteration 2.1 and 2.2 Accuracy comparison***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it.2.2_loss.png)

***Figure 5.11: Iteration 2.2 Loss***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.loss_comp.png)


***Figure 5.12: Iteration 2.1 and 2.2 Loss comparison***
![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_loss_per_step.png)

***Figure 5.13: Iteration 2.2 Loss_per_step***


![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2_loss_per_step.png)

***Figure 5.14: Iteration 2.1 and 2.2 Loss_per_step comparison***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_no_event_accuracy.png)
***Figure 5.15: Iteration 2.2 no_event_accuracy***

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2_no_event_accuracy_comp.png)

***Figure 5.16: Iteration 2.1 and 2.2 no_event_accuracy comparison***

The no_event_accuracy metric shows an interesting development. Appearing to converge around 0.966 for iteration 2.1, it also both improves by a step change and appears to reach a higher level of accuracy, ending at 0.9901 (Figure 5.16). This is likely due both to the pretraining done during iteration 2.1, and to the fact that the style specific sample of 530 is dramatically smaller than the general dataset of 170k. This makes it easier for the model to correctly predict the rhytmic aspects of the music. 

***Output evaluations**
 

*One Note*

The one note sequences generated by iteration 2.2 are more coherent than the ones generated by previous models. They include repetition of local structures such as short two or three note motifs, and also use of local jazz structures such as approaching a target note first from above, then below before hitting the note (MIDI 12). The examples are also quite more rhythmically consistent and developed than in earlieriterations. This suggest the model has learned important aspects of jazz phrasing from the example data. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_onenote_example.png)
***MIDI 12: It 2.2 onenote example***

*Major*

The major melody still includes several stepwise movements, however, the examples are less consistent in terms of following the key and rhytm of the original primer. Instead, it includes occasional quicker rhythms and some chromatic movements. These are characteristics typical for jazz solos, suggesting the model has been somewhat overfit towards this data. 
 
![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_major_example.png)
***MIDI 13: It 2.2 major example***


*Minor*
The minor primer sequences include both stepwise melody movements, and repeated local structures such as stepwise movements as small motifs transposed downwards. The best of the melodies included a developing melody, following repeating local patters to create a global structure (MIDI 14). However, the failure mode of continously repeated notes was also present. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_minor_example.png)
***MIDI 14: It 2.2 minor example***


*Arpeggiated** 

The on the arpeggiated primer iteration 2.2 performed worse than iteration 2.1, failing to capture the local strcture of the primer. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_arp_example.png)
***MIDI 14: It 2.2 arp example***

*Bach** 

ITeration 2.2 also performed worse than iteration 2.1 on the bach examples, capturing fewer of the characteristic interval leaps. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_bach_example.png)
***MIDI 15: It 2.2 Bach example***


*Modal*

However, the 2.2 model outperformed Iteration 2.1 on the modal primer, which was the designed to capture the context specific application. It was able to capture both the rhytm combining short and long notes, and the dorian scale of the primer.  

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/it_2.2_modal_example.png)
***MIDI 15: It 2.2 Modal example***



# Conclusion 


The project successfully met its goals of examning how to effectively generate music melodies that capture the characteristics of a given melody sequence both in general and specific music contexts. Three LSTM RNN  models were developed to this end, Iteration 1 trained on a musical dataset of 11 133 examples, Iteration 2.1: A LSTM RNN trained on a larger general music dataset of 179 781 musical examples and including captured performances, and Iteration 2.2: A LSTM RNN applying transfer learning by continuing training from 2.1 on a small selection of 530 context specific music examples. 

**Final Numerical Results**

| **Model** | **Step** |***Accuracy*** |***Loss***| ***Loss_per_step***| ***no_event_accuracy*** |
| --- |--- | --- | --- | --- | --- |
| Iteration 1| 20k |  0.7347|  0.8976 | 0.8976 | 0.9678 |
| Iteration 2.1 | 19.99k |0.7275 | 0.9203 |0.9203  | 0.9663 |
| Iteration 2.2 | 72.05k | 0.9012| 0.3132 | 0.3132 |0.9901|

The final numerical evaluations show that ITeratin 2.2 managed to achieve the best metric scores of the models, with Accuracy of 0.9012, Loss and Loss per step of 0.3132 and no_event_accuracy of 0.9901. Comparison of the numerical evaluation results also revealed a correlation between dataset size and performance, with iteration 1 performing slightly better than 2.1, and the most specialised iteration 2.2 significantly outperforming both. 

**Musical Output Evaluations**

The general models Iteration 1 and 2.1 are able to generate sequences which capture several key musical characteristics of the primer melodies. It is also found that training on a larger dataset improves model performance, as the output examples generated by 2.2 generally capture musical structures better than the ones generated by 2.1. Model 2.2 is able to generate sequences with several of the musical characteristics of the context specific music data, and performs better than iteration 2.1 on both the onenote primer and the modal primer. The model does however, does lose some of its generality, performing worse than 2.1 on the other primers which are designed to test the models general applicability. This validates the model as a specialised context specific model, and illustrates the difference between general and context specific sequence generation models. 

# Directions for further research: Next Steps in Model and Application Development 

## Tuning the RNN with Deep Q Reinforcement Learning 

Next steps involve an Implementation of the Magenta Deep Q Learning Music model created by Jaques et. al.(2017) using the authors own dataset as described above and customized musical theory rules in the reward function. This would implement the following customized music theory rewards: 

## Customized Music Theory Rewards 

The reward_music_theory function defined in the RL Tuner model (Google AI, 2019d)  individual music theory subfunctions to compute a reward for desired musical outcomes. Among the default music theory rewards include playing a motif, repeating  a motif and following a certain scale. 

The original RL tuner model implemented musical theory rules based on some music domain knowledge and the musical treatise "A Practical Approach to Eighteenth-Century Counterpoint” by Robert Gauldin. In their paper and associated blog posts, Jaques et. al. (2017) encourage further exploration and customization of these music theory rules. 

For the present model these music theory rules can be augmented to suit the specific application context of the model. This can include: 

- Modal classical theory  (Persichetti, 1961)
- Jazz theory (Pease & Pulling, 2001; Pease, 2004; Miller, 1996) 
- Film music Analysis (e.g Adams, 2010).

The original music theory rewards for the DQN RL tuner model only defined a C major scale (Jaques et. al.,2016). The next iteration will therefore include a Modal framework implementing relative scale pitches. These are based on the major modes of western music, which are used both in classical composition (Persichetti, 1061 ), Jazz Composition (Pease & Pulling, 2001; Pease, 2004; Miller, 1996) and cinematic music (Adams, 2010) 

### Online Adaptivity

The next iteration could include the system learning to adapt to a certain performers style using online reinforcement learning. Using an online implementation of the RL tuner, custom rewards could be defined as follows: 


**Rule**: 	Reward when desirable musical outcomes happen
**Desirable musical outcomes**

*Following the melody of the performer*

paralell movements approximate delta(Pt-1, Pt)=  delta (At-1,At) 	
countermelodies approx delta(Pt-1, Pt) = - delta(At-1, At)	
paralell following approx delta(Pt-1, Pt) = delta(At, At+1 --> lead to mimick?  	
paralell counter approx delta(Pt-1, Pt) = - delta(At, At+1 --> lead to mimick?  	
	
 
*Harmonizing with the performer**

harmony: third below A = P-3rd  -(oct)	
harmony: third above A = P + 3rd  -(oct)	
harmony: fourth below  A = P + 3rd  -(oct)	
harmony: fith below A = P - 5th -(oct)	
harmony: 9th below A = P - 9th (-oct)	



## References
- 8 notes (2019).Bach - Cello Suite No.1 in G major, BWV 1007 (complete) midi file for Cello (midi). 8 Notes.  https://www.8notes.com/scores/14093.asp?ftype=midi [Accessed 18.04.2018]
- Abolafia, D. (2016). A Recurrent Neural Network Music Tutorial. *Google AI Magenta Blog.* https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial [Last Accessed 28.04.2019]
- Adams, D. (2010) *The Music of the Lord of The rings Films* Carpentier and Alfred Music Publishing
- Acroche2 Studio (2019).Jazz Midi Files. *Acroche2 Studio Online* http://www.acroche2.com/midi_jazz.html
- AIVA (2019). *AIVA - The Artificial Intelligence composing emotional soundtrack music* https://www.aiva.ai/   [Last Accessed 30.04.2019]
- Conneau, A., Kiela, Douwe, Barrault, Loic, Schwenk, H., & Bordes, A. 2016  Supervised Learning of Universal Sentence Representations from Natural Language Inference Data. *arXiv Online* https://arxiv.org/pdf/1705.02364.pdf  [Accessed 29.04.2019]
- Eck & Schmidhuber. (2002) Finding temporal structure in music: Blues improvisation with LSTM recur- rent networks. In *Neural Networks for Signal Processing*, pp. 747–756. IEEE, 2002
- Fernández, J.D & Vico, F. 2013. AI Methods in Algorithmic Composition: A Comprehensive Survey. *Journal of Artificial Intelligence Research* 48 (2013) 513-582
- Franklin, J.A. 2019. Recurrent Neural Networks for Music Computation. *INFORMS Journal on Computing* 18(3):321-338. https://doi.org/10.1287/ijoc.1050.0131
- Géron, A. (2017) *Hands-On Machine Learning with Scikit-Learn & Tensorflow* O´Reilly Media Inc, Sebastopol.
- Géron, 2019. Chapter 4: Recurrent Neural Networks. *O´Reilly Library* https://www.oreilly.com/library/view/neural-networks-and/9781492037354/ch04.html [Accessed 05.05.2019]
- Giddins, G & Devaux, S. (2009) *Jazz*. Boston, Berklee Press. 
- Goldstein, G. (1982)  *Jazz Composer´s Companion* Boston, Berklee Press.
- Google AI (2019). Magenta - Make music and Art Using Machine LEarning. https://magenta.tensorflow.org/[Accessed 19.04.2018]
- Google AI Magenta (2019a). Magenta Github Repository. https://github.com/tensorflow/magenta [Accessed 19.04.2018]
- Google AI Magenta (2019b). The Maestro Dataset. *Tensorflow Magenta* https://magenta.tensorflow.org/datasets/maestro
- Google AI MAgenta (2019c). Melody RNN. *Magenta Github Repository* https://github.com/tensorflow/magenta/tree/master/magenta/models/melody_rnn 
- Google AI Magenta (2019d). RL Tuner. *Magenta Github Repository* https://github.com/tensorflow/magenta/tree/master/magenta/models/rl_tuner 
- Goodfellow, I., Bengio, Y., & Courville., A.(2016) *Deep Learning* MIT Press, London. 
- Graves, A (2013). Generating sequences with recurrent neural networks. *arXiv preprint:1308.0850.*
- Hochreiter, S. & Schmidhuber, J (1997). Long short-term Memory. *Neural Computation* 9(8), 1735-1780. 
- Jaques, N 2016. Tuning Recurrent Neural Networks with Reinforcement Learning. *Google AI Magenta Blog* https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning  [Accessed 27.04.2019]
- Jaques, N., Gu,S., Turner, R E., & Eck, D. 2017.'TUNING RECURRENT NEURAL NETWORKS WITH REINFORCEMENT LEARNING'.  NIPS 2016 -Deep Reinforcement Learning Workshop. Available at: https://arxiv.org/pdf/1611.02796v2.pdf [Accessed 04.04.2019]
- Jarrahi, M.H (2018) Artificial intelligence and the future of work: Human-AI symbiosis in organizational decision making. Business Horizons. No. 61 pp. 577-586. 
- midiworld (2019a). Movie Themes. *Midiworld Online* http://www.midiworld.com/search/1/?q=movie%20themes  [Last Accessed 29.04.2019]
 - midiworld(2019b). Video Game Themes. *Midiworld Online* http://www.midiworld.com/search/?q=video%20game%20themes [Last Accessed 29.04.2019]
 - midiworld (2019c). TV Themes. *Midiworld Online* http://www.midiworld.com/search/?q=tv%20themes  [Last Accessed 29.04.2019]
- midiworld. (2019d). Classic. *Midiworld Online*  http://www.midiworld.com/classic.htm#d [Last Accessed 29.04.2019]
- Mikolov et al.(2010) Recurrent neural network based language model. *In Interspeech*, volume 2, pp. 3.
- Miller, Ron. 1996. *Modal Jazz Composition & Harmony vol. 1.* Rottenberg: Advance Music
- Miller, Ron. 2000. *Model Jazz Composition & Harmony vol.2* Rottenber: Advance Music
- Parry, K. Cohen, M & Bhattacharya, S (2016) Rise of the Machines: A Critical Consideration of Automated Leadership Decision Making in Organisations. Group and Organisation Management. 2016 vol. Vol. 41(5) pp. 571–594
- Pease, T. & Pulling, K. 2001 *Modern Jazz Voicings* Berklee Press, Boston MA.
- Pease, T. 2004 *Jazz Composition: Theory and Practice* Berklee Press, Boston MA. 
- Raffel, Colin. (2019). *Lakh Midi Dataset v0.1*. https://colinraffel.com/projects/lmd/ [Accessed 19.04.2019]
- Raffel, Colin.2016 "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". *PhD Thesis*. 
- Russel, S & Norvig, P. (2016) *Artificial Intelligence - A Modern Approach*. 3rd Edition. Pearson Education Limited, Essex. 
-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
- Persichetti, V. 1961. *Twentieth Century Harmony* New York: W.W Norton & Company. 
- The Jazzomat Research Project (2019). Data Base Content.*Jazzomat research project Online*  https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html  [Last Accessed 29.04.2019]
- The MIDI Association, 2019. The Complete MIDI 1.0 Detailed Specification. *The MIDI association Online* https://www.midi.org/specifications-old/item/the-midi-1-0-specification [Lasr Accessed 29.04.2019]
- Zoph, B., Yuret, D., May, J & Knight., K . 2016. Transfer Learning for Low-Resource Neural Machine Translation. arXiv Online. https://arxiv.org/pdf/1604.02201.pdf [Accessed 29.04.2019]
