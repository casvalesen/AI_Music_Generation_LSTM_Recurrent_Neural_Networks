# Project Report: Music Generation with AI and Deep Learning (WIP) - Domain specific Sequence Modelling 

### Project Criteria: 

Individual project on a topic falling in the scope of the course
• Focus on methodological aspects, principles, and implementation

Allowing for
• Accounting for individual preferences
• Allowing to explore a given topic in more depth
• Allowing to gain hands-on experience in coding a solution

Selected:
• Implement and evaluate a neural network for a sequence modelling task

**refer to specific code modules, show implementation understanding**

# Introduction and Research Question

## Project Overview 


The project models entail customized implmentations of the Google AI Magenta models for melody generating using a LSTM Recurrent Neural Network (Google AI Magenta, 2019c) 

(and a Reinforcement Learning algorithm Tuning this network using Deep Q-Learning (Google AI Magenta, 2019d; Jaques et. al., 2017). )

Dataset choice, Model choice parameter options and customizations are informed by the requirements of the application context, application music practitioner literature, the authors own domain knowledge from 18 years as a music practitioner. 

Supervised learning 

*The project includes:* 

- Dataset building for general sequence modelling in musical domains
- Dataset building for sequence modelling in the specific application context. 
- Training a RNN LSTM for general sequence prediction
- Applying learning to further train the Model  towards a context domain specific application
- Empirical evaluations of trainig iterations in transfer learning.
(- Using Deep Q Reinforcement Learning to tune this context specific model further using customized rewards from the application domain. )
- Design and application of context specific model evaluations mechanisms informed by application domain literature and practice. 

**Summary of Results**

## Project Goals

### Application Goals

The Goal is to design a music generator that is capable of generating melodies that are both musically expressive 
and have a certain coherency. The model will generate a monophonic melody line, (which will be responsive to a real time input melody. )

### Research Goals / Question 

Thus, the project entails the implementation and evaluation of a neural network for a sequence modelling task in the context of music generation. IT was basedbased on the following research question:

*How to effectively music melodies that are able to capture and imitate the characteristics of a given melody sequence both in general and specific music style contexts.* 

This was examined using the following models: 

-  *Iteration 1: A LSTM RNN model trained on a general musical dataset of 11 133 musical examples.* 

- *Iteration 2.1: A LSTM RNN trained on a larger general music dataset of ( ) musical examples and including captured performances* 

- *Iteration 2.2: A LSTM RNN applying transfer learning by continuing training from 2.1 on a smaller selection of context specific music examples. 

## Artificial Intelligence Research Context 

As with text and visual art generation, music generation falls within the space of computational creativity (Eck & Schmidhuber, 2002; Franklin, 2009)   In addition to providing a number of usecases, it is also interesting in the sense that creativity is a fundamental aspect of human intelligence (Russel & Norvig, 2016). It is recognized that contemporary AI hold advantages over humans in analytical tasks such as identifying relationships and caputuring complex dynamics of large datasets. However, creativity is argued to be one of the areas in which humans still hold significant (Parry et. al, 2016 p.580; Jarrahi, 2018 p.582). Exploration of computational creativity is therefore an important step on the path to general or strong Artificial Intelligence (Russel & Norvig, 2016 p.16; Jaques et. al.,2017 ). 

## Application Context

The application context of the music project is in the realm of generating melodies for real time music improvisations in a cinematic style, such as film and game music. 

# Choice and Description of Data 

## Musical Data & Patterns

**Midi format** 
Similar to earlier research (Franklin, 2019), this project focuses on digital music at the pitch and duration level. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/bach_notes.png) 
**Figure 1: Musical Notation of Melody, Bach Cello Suite 1**

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/bach_midi.png) 
**Figure 2: Midi Notation of Melody, Bach Cello Suite 1**


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

The data consisted of midi files and was selected with the overall application. 
- Building custom dataset
- Transform to melodies from 
- Think about target and prediction

Midi files were inspected using the musical notation software Sibelius. 
midiworld(2019d) 

Theoretically, this is similar to training approaches used in Natural Language processing for building language context specific models (Reference needed). 

Style extraction? 

### General melody generation

- General melody dataset: Lakh dataset 178 579 midi files containing    (Lakh

- Performance: Maestro, 1 202 midi files capturing world class performers classical musical material and performance 

### Style Specific melody generation 

Style specific dataset. 530 files combining both polyphonic and monophonic instances. 

**Movie themes:**  A subset of movie themes. Themes of a symphonic nature were emphasized (midiworld, 2019a) 

**Video Game themes:** A subset of Video Game themes (midiworld, 2019b) 

**TV Themes:** A subset of TV themes (midiworld (2019c)

**Modal Classical:** Classical music from selected composers were also included as part of the style specific dataset. This was composers who were in the expressionist, impressionist and nasjonal romantic with inspiration of Scandinavian folk music (..) paradigms as this was the moderns sounding style that would suit the model context (Persichetti, 1961) 

**Jazz Data:** The purpose of including the jazz solos was for the model to learn jazz phrasing and performance, with specific focus on improvsational melody construction. Most jazz material is written down only in sparse form, with practitioners given ample freedom to improvise and interpret hte material. Most practitioners therefore learn by imitating and transcribign recordings rather than playing after sheet music .  This tanscription of individual performances is a slow and incredibly time consuming task, which practitioners spend years perfectin. Books are released with transcriptions of individual performances, but these are very rarely in midi format (Giddins & Devaux, 2009). For capturing sufficient data sets to train AI models it is therefore nessecary to source transcribed midi jazz from the web, and the material for the jazz part of the dataset was sources from The Jazzomat Researhc Project (2019) and Acroche2 Studio (2019). A selection of 64 jazz solos and 58 ensemble pieces and where included in the the dataset based on the artists emphasis on modal melodies and modern harmonic colours. These were solos by Miles Davis, John Coltrane, Herbie Hancock, Wayne ShorterKenny Wheeler and Weather Report (Giddins & Devaux, 2009).  

# Solution Concepts 

## Models

The following models were created: 

- Model 1: A Note based LSTM RNN for General Music Generation trained on a subset of the General Music Dataset, only including musical composition.
- Model 2.1 : A Note Based LSTM RNN for General Music Generaton trained on the entire General Music Daset, including both musical compositions and captured performances. 
- Model 2.2: A Note Based LSTM RNN which applied transfer learning to adapt Model 2.1 to the application context domain by training it on the style specific dataset. 
(- Model 3.0: A Deep Q Reinforcement Learning model which tunes model 2.2 by balancing style specific music theory rewards with the LSTM based sequence model. ) 
- Model   (Custom RNN) ) 
 

### LSTM Recurrent Neural Networks for sequence generation 


The context application and research question entailed how to generate a monophonic, aka. single voice musical melody capturing and imitating the characteristics of a given melody in general and specific music contexts. In AI terms this entails generating a sequence of states which capture and immitate the characteritics of a given sequence of states. Music generation can considered a dynamic system, aka. a feedback proces, since the current state is dependent on a history of past states (Franklin, 2019).

Applying the classical form of a dynamical system (Goodfellow et. al., 2016), a music system can be considered as: 


<img src="http://latex.codecogs.com/svg.latex?s^t=f(s^{t-1}; \theta)" border="4"/>

with <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/> being the current state of the music. 

![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/dynamical_music.png)
**Figure 3 : Melody as dynamical system, musical notation** 
![alt text](https://github.com/lse-st449/st449-projects-casvalesen/blob/master/pictures/dynamical_music_midi.png)
**Figure 4 : Melody as dynamical system, midi data** 

Figure 3 and 4 illustrate the monophonic melody case. <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/> correspond to the current note while  <img src="http://latex.codecogs.com/svg.latex?s^{t-1}" border="0"/> corresponds to the preceeding note.  <img src="http://latex.codecogs.com/svg.latex? \theta" border="0"/> is applied to all time steps , incorporates information about the whole sequence (blue). In the above example, this could be used to infer the note at  <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/>  by including information that the starting note repeats after 8 time steps(red line). It could also capture that the melody consists of two alternating unique patters of 4 values (purple and yellow lines), with the pattern beginning at <img src="http://latex.codecogs.com/svg.latex?s^t" border="0"/>  has the same start note value as the melody starting note.  


This makes music generation a sequence modelling task that involves generating a sequence of outputs given a sequence history of inputs.  Sequence2Sequence Recurrent Neural networks are therefore good model candidates (Géron, 2017 p.407; Goodfellow, Bengio & Courville, 2016). While Feed Forward and Convolutional Neural Networks generate a fixed size output from a fixed size input, Recurrent Neural Networks are able to generate output sequences of arbitrary length from input sequences of arbitrary length (Géron, 2017 p.407). This also supports the use of an RNN model for creating a music generation model responsive to improvisation, as improvised musical phrases often vary in lenght (Giddins & Deveaux, 2009). 


**LSTM for Music Generation** 


A challenge in Recurrent Neural Networks is that the gradients tend to either vanish or explode when they are propagated over many stages. This property makes it hard for regular RNN models to learn long term dependencies (e.g. Goodfellow et. al., 2016 p. 390). An essential feature of most musical improvisations and compositions is the repetition, developments and selfreference of musical patterns over a high number of sequence steps (Giddins & Deveaux, 2009; Persichetti, 1961; Pease, 2004). In order to perform well, musical generation models must therefore address this challenge. Indeed, research has found that models composed of standard Recurrent Neural Nets fail to generate music with global coherence (Eck et. al., 2002). 

Problem of long term dependencies 
Long term dependenceis 
LSTM solves 

(Hochreiter & Schmidhuber, 1997)
LSTM computations adapted from Géron (2017 p. 409)


In the research literature, LSTM Recurrent Networks have been used to generate (....) and Improvisation (Eck & Schmidhuber, 2002; Franklin, 2019). Considering the end goal of the current application is also in the realm of improvisation, with a subset of the data coming from Jazz, the latter two papers make the LSTM RNN model especially relevant. 

(Math formula) - Latex 

<img src="http://latex.codecogs.com/svg.latex?f" border="0"/>

(Graphic)


LSTM musical setting, how 
Goodfellow, I., Bengio, Y., & Courville., A.(2016 p.397)
Seq2Seq model 
*Character RNN*  predict one data point at a time, with discrete values (Graves, 2013)
Goodfellow, I., Bengio, Y., & Courville., A.(2016 p.397)
Géron, A. (2017 p.407)
Magenta Melody RNN (Abolafia, 2016; Google AI Magenta, 2019c)

### Model Specifications 

**Training Input**

A sequence of pitches over 128 possible midi values. For a given training melody sequence at step t, the input sequence value at time step *t* corresponds to sequence entry *n*.  

**Training Labels** 

A sequence of pitches over 128 possible midi values, with each sequence value corresponding to the next step sequence value of the input sequence. For a given training melody sequence , the input sequence value at time step *t*  thus corresponds to sequence entry *n+1*.   

**Generation input: Primer**

An arbirary length sequence of pitches over 128 possible midi values  

**Generation Outputs**

An arbitrary length sequence of pitches over 128 possible midi values, capturing primer sequence pattern characteristics, i.e musical attributes. The precise length is specified in the model parameters.  


### General Music Model

As with Language models, a music model can be first trained as a general model, then adapted to the specific application context. In Language this could entail building a general model for the language in question, then specialising the model to a certain application field using application specific data (*NLP transfer learning reference) 

In music, this training process could be utilised by first training the model on a general music data from which the model could learn general musical rules and conventions. The model could then be further trained on a data from specific style of music in order to specialise it to this field. 

In the first iteration, the model will consist of a music generation RNN LSTM. 

This is uses a similar basis as character level RNN´s developed for Natural Language Processing, and tries to predict the next token given a sequence of previous values  (Abolafia, 2016; Jaques et. al.,2017; Mikolov et. al.,2010) In the RNN LSTM, the token used for prediction is known as the "primer" for the model (Abolafia, 2016; Google AI Magenta, 2019c)


### Style Specific Model

Transfer learning was applied to train the style specific model. This approach entails using a model trained on a general dataset as starting point for further training of a model on a specific dataset. This is in order to make the second model more general, and it is build on the assumption that many of the factors explaning variations in the first dataset hold in the second.  (Goodfellow et. al.,2016). Both general music data and the context specific data are governed by the basic musical building blocks described in the Musical Data setion (Persichetti, 1961; Pease & Pulling, 2001). These musical building blocks are analogous to the low level visual features captured in CNN visual systems, where transfer learning is frequently applied to adapt models trained on one visual category to another (Goodfellow et. al., 2016). 


(This is confirmed by examination of both datasets, and th The general music data contains a distribution over all types and types of music, while the context dataset was aiming at a very specific syle. The second iteration was then trained on a style specific dataset to approach the musical style of the application domain. ) 

# Model Implementation 

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

**Gdrive integration**  

The preprocessing was done on a local hardrive. The data was then uploaded to gdrive, with was mounted to Colab using the standard Colab mount function: 
```
#Mount drive to Google Colab 
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)   #/content/gdrive/My Drive/AI_music/ , force_remount=True
root_path = 'gdrive/My Drive/AI_music/'

```
This ensured files could be called directly from the colab and all processing could be cloud based. 

***Model Building and Training** 

The Magenta distribution (Google Magenta AI, 2019c) supports commmand line training of RNN based models. However, to build a customized model using the library was necessary to thoroughly comprehend and apply sections of the Magenta distribution source code. The Melody RNN Model  distribution consists of eight interlinked python modules: 

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

Several of the subfunctions take ```unused_argv```as inputs. After a visit to stackoverflow, it was apparent this is internal code practice at Google, and the variable can be set to     ``` unused_argv= ' '``` to avoid any issues. 


***Custom generate_test_melodies() function*** 

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

  FLAGS.output_dir=output_dir+'arpeggiated'  #/content/gdrive/
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

### Custom functionality 

Evaluation function 


 - FLAGS 

### TensorBoard

The data was inspected using tensorboard. 

- Custom function to 


(### DQN Reinforcement Learning 

The third iteration will then tune this LSTM using the Deep Q-learning approach by Jaques et. al.(2017).  ) 

# Numerical Evaluation 

## Evalation mechanisms 

Standard neural network performance metrics were used to evaluate the models: 

**Accuracy** The ratio of correctly predicted values  (Géron, 2017). 

**loss_per_step** The amount of loss for each training step. The melody rnn model defines this as the Mean Softmax cross-entropy loss, implementing the loss function ```tf.nn.sparse_softmax_cross_entropy_with_logits```. This loss function trains model to estimate high probabilities for the target class by penalizing low target class probabilities (Géron, 2017 p.143). 

- (function)- Mean - Softmax - Cross entropy with logits     (Goodfellow et. al.,2016 p. 73; Géron, 2017 p.143) 

**no_event_accuracy** The ```MELODY_NO_EVENT``` indicator is implemented in the Magenta library to mark when a sustained state is held over several values, which means the current status of ```NOTE_OFF``` or ```NOTE_ON``` is kept. In the library source code, ```No_event_accuracy``` is defined as the sum of the products of correct_predicitions and no_events, divided by the sum of no events. 

```
   no_event_accuracy = (
            tf.reduce_sum(correct_predictions * no_event_positions) /
            tf.reduce_sum(no_event_positions))
```
(Google AI MAgenta, 2019c)

This is therefore a music specific numerical evaluation which gives a measure of how accurate the value durations, aka. rhytms, sequences are compared compared to the target sequences. 



### Stats Tables




### Domain Evaluation 
*Reference on how NLP models are evaluated**

As a creative RNN is in its nature not trying to predict an accurate value such as a time series or classification, 
evaluation of creative RNN is in its nature 

Although accuracy rate might give an objective measure of how well the model predicts a melody note, it is not very informative on how the model performs in context specific terms, namely in how it manages to be musically expressive and use established musical conventions. 

Criteria: 

Analysis: A musical analysis. Context specific. 

STATS?? 

- It was therefore needed to define 

- Approaches to evaluation taken in papers, with refs (.....). Magenta refs, Jaques et. al., other papers for magenta.

This model evaluation definitions are in the field of explainable AI( ref). Since the objective measures or parameters captured by the model are not necessarily informative in the application context in the same way coefficients would be in regression, it is necessary to probe the AI model to generate outputs to get a general sense of how the model is performing and reasoning in the subject specific context (ref). 

- Note on explainable AI. 

**General** 

A known failure mode of single step prediction RNN´s in sequence generation is the continuous repetition of the same token(Jaques et. al., 2017). The results of the LSTM RNN  were therefore evaluated on whether this failure mode was occuring. 

Another common failure mode of single step RNNs is the difficulty of ensuring coherent global structure. In music specifically, this global structure is built up by musical phrases (Jaques et. al., 2017). An evaluation criteria for the generated sequences is therefore to what extent they have a coherent global structure.

In a musical context this can refer to the following characteristics: 


### General Model Evaluation


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


## Data processing 

### Handling of Musical Data

- Midi format: Basic properties of musical data
Like text data, musical data in fundamentally sequential. 
Pitch at timestep 
Duration interpreted as on/off at timestep 

- The midi data was converted to the NoteSequence protocol developed by Google Magenta (Reference).   

(Graph of bach cello suite). 

###  Working with tf Records

- Refer to command line code for data processing 


(### Preliminary Experiment: Bach Bot - Monophonic Cello RNN 

In order to familiarize myself with the Google Magenta toolkit and create a simple baseline model for further iterations, I first trained a simple model on using data from bachs cello suites. The advantage of this data is that it is mostly monophonic, e.g one melody, and captures many aspects of classical melodic and harmonic movement such as development of a repeated theme, development of a harmonic sequence and dramatic development. In data science terms, this means the data has a includes data patterns common in western musical traditions, such as absolute value differences in the sequence (melodic intervals), relative value difference patterns that are repeated with different absolute values locally (thematic development), and meta patterns  for how these different types of relative value difference patterns alternate depending on their absolute values (harmonic development). 

- Musical Analysis of bach cello suite in data science language

The initial cello suite was downloaded from an open source midi file repository (8 Notes, 2019). 

- Using default Magenta Tensorflow graphs )

**Output Naming Convention:** it_(iteration_nr)_(primer)_nr_(outputnr)_(training_ep if applicable)

## Model Iteration 1: Mono_rnn based on Magenta´s RNN LSTM Model 

- First part of Lakh dataset used, ... entries. 

- input data: 130 , 128 pitches on/off, 2 values for ....?

- Polyphonic midi files, aka. multiple simultaneous. Preprocessing extracted individual melodies / melody segments. 

- Mono_rnn
- RNN lstm model, 

- Lakh only?

**Model Outputs & Evaluation**


***Accuracy***

***Loss_per_step***

***no_event_accuracy***

E.g 

Key? 
intervals? 
harmonic development? 

*Primer: One note* 

- (midi/ audio example) 

*Primer: Simple major Melody* 

- (midi/ audio example)

*Primer: Simple minor Melody*

- (midi/ audio example)

*Primer: Arpeggiated chord progression*


- (midi/ audio example)

*Primer: Bach Cello Suite 4 bars 

- (midi/ audio example)


## Iteration 2: Mono_rnn based on Magenta´s RNN LSTM Model, complete Lakh dataset (2.1) and style specific dataset (2.2)

**2.1: Complete Lakh and Maestro datasets

**Model Outputs and Evaluation***

***Accuracy***

***Loss_per_step***

***no_event_accuracy***

Diverging results. 


For the second iteration, a sixth primer melody was added. This was a simple phrase which would be similar to what would be used by the musical instrument player in the improvisation application context. 

Lakh dataset: 
Maestro:

*Primer: One note* 

- (midi/ audio example) 

*Primer: Simple major Melody* 

- (midi/ audio example)

*Primer: Simple minor Melody*

- (midi/ audio example)

*Primer: Arpeggiated chord progression*


- (midi/ audio example)

*Primer: Bach Cello Suite 4 bars 

- (midi/ audio example)


*Modal* 

- (midi/ audio example)

It was a modal melody, e.g. a melody following a single 


- Training statistics:  step change in several metric on context specific dataset as this is a lot less varied than the general dataset, thus easier with precision(?)

- Cloud based training terminated, and was restarted. A memory overload issue led to loss of training data between (---). However, in the metric visualisations, this provides a good visualization of the step change occuring between training model 2.1. and 2.2. 


**Iteration 2.2: LSTM RNN Context Specific Model** - Combine for discussion on transfer learning and training. (!!). 

*Training* :  

Transfer Learning (Géron, 2017 p.289)
Goodfellow et. al., 2016 p.

The style specific data contained a much lower number of than the instances in the Lakh and Maestro training sets combined, 530 as opposed to 179 781 instances. The data was selected because it was closer to the instances likely encountered in the application context. 


Overfitting the model on this data as opposed to keeping it balanced between the general and specific datasets was therefore justified. (?)

Initially, the model was overfitted on this data in order to gain context specificity With 50 000 additional training iterations. However, the generated melodies had forgotten: 

THIS IS CRAZY; CHECK!!!(A rather Naive approach was used for this second stage training, in which the training  steps was increased by a factor 179 781/530≈ 340 to give equal weight to the context data. (?)


**Model Outputs & Evaluation**

***Accuracy***

***Loss_per_step***

***no_event_accuracy***


(Paper on transfer learning & Two step training for language models) 

*One Note*

Some more chromatics, shows jazz influence in model. 

- (midi/ audio example)

*Major*

- (midi/ audio example)

*Minor*
stepwise melody movements, stepwise movements as small motifs transposed (min 2.2-1)
failure mode of continuously repeated notes in  (minor 2.2-2)

- (midi/ audio example)

*Bach** 

- (midi/ audio example)
2.1 captured interval leaps better. 

2.1 more general dataset, thus more adaptable. 

Overfit? 

*Modal*
Does it perform better on this specific? 


Generated from previous: 

It would seem the model has lost a lot of its generality. It does perform well on ....  , but some more generality is needed. 
To try to remedy this, the model trained on (...) iterations was used. 
 - This shows  lack of generalisation


**2.2: fewer episodes**

In order to keep the generality of iteration 2.1 with some context specialisation, further melodies were generated using the saved checkpoint 22 158. This version of the model was trained on the initial 20 000 episodes with the large dataset, then 1/10 th of the iterations on the context dataset. 

- (discuss outcomes on models) 
- Link to transfer learning paper (ref)


*Primer: One note* 

- (midi/ audio example) 

*Primer: Simple major Melody* 

- (midi/ audio example)

*Primer: Simple minor Melody*

- (midi/ audio example)

*Primer: Arpeggiated chord progression*


- (midi/ audio example)

*Primer: Bach Cello Suite 4 bars 

- (midi/ audio example)


*Modal* 

- (midi/ audio example)

It was a modal melody, e.g. a melody following a single 

# Tuning the RNN with Deep Q Reinforcement Learning 

Implementation of the Magenta Deep Q Learning Music model created by Jaques et. al.(2017) using the authors own dataset (as described above) and customized musical theory rules in the reward function. 


### Deep Q Reinforcement Learning

Goodfellow et. al., 2016. 
Géron, 2017. 

## Deep Q Learning for Sequence Prediction 

### Magenta DQN Rl Tuner

Exploration mod: "egreedy" implementing epsilon greedy policy or boltzman to sample from its own outputs 
Priming mode: 

## 

## Customized Music Theory Rewards 

The reward_music_theory function defined in the RL Tuner model (Google AI, 2019d)  individual music theory subfunctions to compute a reward for desired musical outcomes. Among the default music theory rewards include playing a motif, repeating  a motif and following a certain scale. 

The original RL tuner model implemented musical theory rules based on some music domain knowledge and the musical treatise "A Practical Approach to Eighteenth-Century Counterpoint” by Robert Gauldin. In their paper and associated blog posts, Jaques et. al. (2017) encourage further exploration and customization of these music theory rules. 

For the present model these music theory rules were augmented to suit the specific application context of the model. 

Modal classical theory: 
Jazz theory: Pease & Pulling, 2001; Pease, 2004; Miller, 1996) 
Film music:  LOTR musical rules for flavour 

The original music theory rewards for the DQN RL tuner model only defined a C major scale. 
(https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_ops.py) 

Modal framework implementing relative scale pitches. These are based on the major modes of western music, which are used both in classical composition ( ) , Jazz Composition (Pease & Pulling, 2001; Pease, 2004; Miller, 1996) and cinematic music (Adams, 2010) 


**Modal Keys** 
(Picture of musical notes for scales) - from composition book. (eg. Miller, 1996)

- (midi trepresentations of the scales)

- details of how they were added, with relative keys. 


# Conclusion 


# Next Steps in Model and Application Development 

**Add interactivity**

- using the .js packages
- max msp, take the system online. 

# Directions for further research 

- Adapting to the player. 


## Data sources
8 notes (2019).Bach - Cello Suite No.1 in G major, BWV 1007 (complete) midi file for Cello (midi). 8 Notes.  https://www.8notes.com/scores/14093.asp?ftype=midi [Accessed 18.04.2018]

http://www.midiworld.com/classic.htm#d

https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html

http://www.acroche2.com/midi_jazz.html

https://groups.google.com/a/tensorflow.org/forum/#!topic/magenta-discuss/6ZLbzTjjpHM

## References
- Abolafia, D. (2016). A Recurrent Neural Network Music Tutorial. *Google AI Magenta Blog.* https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial [Last Accessed 28.04.2019]
- Adams, D. (2010) *The Music of the Lord of The rings Films*
- Acroche2 Studio (2019).Jazz Midi Files. *Acroche2 Studio Online* http://www.acroche2.com/midi_jazz.html
- AIVA (2019). *AIVA - The Artificial Intelligence composing emotional soundtrack music* https://www.aiva.ai/   [Last Accessed 30.04.2019]
- Eck & Schmidhuber. (2002) Finding temporal structure in music: Blues improvisation with LSTM recur- rent networks. In *Neural Networks for Signal Processing*, pp. 747–756. IEEE, 2002
- Franklin, J.A. 2019. Recurrent Neural Networks for Music Computation. *INFORMS Journal on Computing* 18(3):321-338. https://doi.org/10.1287/ijoc.1050.0131
- Géron, A. (2017) *Hands-On Machine Learning with Scikit-Learn & Tensorflow* O´Reilly Media Inc, Sebastopol.
- Giddins, G & Devaux, S. (2009) *Jazz*. 
- Goldstein, G. (1982)  *Jazz Composer´s Companion* 
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
- Russel, S & Norvig, P. (2016) *Artificial Intelligence - A Modern Approach*. 3rd Edition. Pearson Education Limited, Essex. 
-  https://arxiv.org/pdf/1810.12247.pdf
- https://arxiv.org/pdf/1803.05428.pdf
- Persichetti, V. 1961. *Twentieth Century Harmony* New York: W.W Norton & Company. 
- The Jazzomat Research Project (2019). Data Base Content.*Jazzomat research project Online*  https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html  [Last Accessed 29.04.2019]

Cut outs: 


"Music is an interesting test-bed for sequence generation, in that musical compositions adhere to a relatively well-defined set of structural rules. Any beginning music student learns that groups of notes belong to keys, chords follow progressions, and songs have consistent structures made up of musical phrases." (Jaques et. al., 2017). (Online) 

"The music theory rules implemented for the model are only a first attempt, and could easily be improved by someone with musical training."  (Jaques et. al., 2017). (Online) -  Improve music theory rule? 

Music generated my artificial intelligence    (AIVA, ;(game music)).

- Ref. on call for application of domain knowledge 


