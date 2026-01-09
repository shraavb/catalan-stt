-----------------------------------------------------------------------------------------------
PRESENTATION
-----------------------------------------------------------------------------------------------

This README file aims to explain users how the CIEMPIESS Corpus is organized
and what kind of files does it have.

The CIEMPIESS Corpus was created at the Speech Processing Laboratory of the
Faculty of Enegineering (FI) in the National Autonomous University of Mexico (UNAM)
in 2012-2014 by Carlos Daniel Hernández Mena, supervised by José Abel Herrera Camacho,
head of Laboratory.

CIEMPIESS is the acronym for:

"Corpus de Investigación en Español de México del Posgrado de Ingeniería Eléctrica 
y Servicio Social".

The CIEMPIESS is a Radio Corpus that was mainly designed to create acoustic models for automatic 
speech recognition and it is made up by recordings of spontaneous conversations between a 
radio moderator and his guests. 

These recordings were taken in mp3 from "PODCAST UNAM" (http://podcast.unam.mx/) and they 
were created by "RADIO-IUS" (http://www.derecho.unam.mx/cultura-juridica/radio.php) that is 
a radio station that belongs to UNAM.

For more information and documentation see the CIEMPIESS-UNAM Project website at:

		http://www.ciempiess.org/

-----------------------------------------------------------------------------------------------
TERMS OF USE
-----------------------------------------------------------------------------------------------

CIEMPIESS Corpus by Carlos Daniel Hernández Mena is licensed under a 
Creative Commons Attribution-ShareAlike 4.0 International License. 
To view a copy of this license visit http://creativecommons.org/licenses/by-sa/4.0/. 
Based on a work at http://odin.fi-b.unam.mx/CIEMPIESS-UNAM/.


-----------------------------------------------------------------------------------------------
GENERAL ORGANIZATION OF THE DIRECTORIES
-----------------------------------------------------------------------------------------------

The CIEMPIESS directory contains the following directories:

	- transcriptions

	- train

	- test

	- textgrids

	- sphinx_experiments

and the following files:


	- LICENSE.txt

	- README.txt

The following is a detailed explanation of the files in every directory.

-----------------------------------------------------------------------------------------------
NOTICE THAT
-----------------------------------------------------------------------------------------------

The design of the CIEMPIESS corpus was very influenced by the CMU-SPHINX3 Speech Recognition
Software.

That is why you can find a directory named "sphinx_experiments" and in general, all the files
of the CIEMPIESS are made to match with the format of the configuration files of the SPHINX3.


Maybe you might see this online tutorial to understand with more detail our influences
in the design of the CIEMPIESS:

		http://www.speech.cs.cmu.edu/sphinx/tutorial.html


-----------------------------------------------------------------------------------------------
"transcriptions" DIRECTORY
-----------------------------------------------------------------------------------------------

You will find the following files that come in pairs:

	- CIEMPIESS_FULL_TRAIN.transcription
	- CIEMPIESS_FULL_TRAIN.fileids

	- CIEMPIESS_test.transcription
	- CIEMPIESS_test.fileids

	- CIEMPIESS_train.transcription
	- CIEMPIESS_train.fileids


The "FULL_TRAIN" files require an historical explanation:

When the first version of the CIEMPIESS was finished on December 2013, it contained
a total of 16717 audio files with their transcriptions.

You can always identify these "primitive" audio files because they have identification 
keys like:

	0173M_09ALX_22OCT12
	0177M_09ALX_22OCT12
	0020M_11ALX_10DIC12
	0001F_12ALX_17DIC12
	0009F_12ALX_17DIC12

(Where F is for "Female" and M is for "Male" voices)

After that, it was decided that two audio sets must be selected for "train" and "test" 
stages. So we took a total of 700 files of these "primitive" audio files and then added 
another 300 with different identification key format, for example:

	F09MAY1844_0036
	F09MAY1844_0038
	AB01_1
	AB01_4
	OSC_001
	OSC_003

These 300 audio files come from different sources and they are prior to the creation of 
the CIEMPIESS.

At the end, the CIEMPIESS is divided in "train" and "test" sets, and you can manage these
sets with the corresponding "_train" or "_test" files in the "transcriptions" directory,
but if you want to work with only the "primitive" files of the CIEMPIESS, you have to
choose the "FULL_TRAIN" files instead.

-----------------------------------------------------------------------------------------------
"train" DIRECTORY
-----------------------------------------------------------------------------------------------

You will find the following directories:

	- ALX_TRAIN

	- ANG_TRAIN

	- MAB_TRAIN


The "primitive" audio files of the CIEMPIESS were extracted by three different workers:
Alejandro (ALX), Angel (ANG) and Mabel (MAB) and that is the reason for naming
these three directories the way they are.

Inside each of them, you will find a set of recordings you may need for performing
a training stage with SPHINX3.

All of the files in the "train" directory has the same identification key format, that is:

		                       0001M_01ALX_17DIC12

         0001			        M_	          01ALX_	              17DIC12

  A relative number            Gender of the Speaker      This is the             This is the date
that identifies one certain        "M" for Male        "01" directory of           when the entire
file inside a directory           "F" for Female       the "ALX" recordings     directory was created


-----------------------------------------------------------------------------------------------
"test" DIRECTORY
-----------------------------------------------------------------------------------------------

You will find the following directories:

	- ciempiess

	- description

	- fm

	- read


As previously mentioned, the "test" set comes from different sources:


ciempiess: Here we have the 700 "primitive" audio files extracted from the first version
           of the CIEMPIESS. All of these files have the identification key format
           shown above (see the section: "train" DIRECTORY).

description: This directory contains 200 recordings of spontaneous speech of people describing
             paintings or answering questions.

fm : This directory contains 17 recordings extracted from the FM Radio. The radio station
     selected for these recordings is different of the radio estations selected to create
     the CIEMPIESS.

read: This directory contains recordings of read speech. The speaker in these recordings
      is a male person between 25 and 30 years who has lived in Mexico City all his life.

-----------------------------------------------------------------------------------------------
"textgrids" DIRECTORY
-----------------------------------------------------------------------------------------------

You will find the following directory:

	- full_train

and the following text files:

	- CIEMPIESS_FULL_TRAIN.label_transcriptions
	- CIEMPIESS_FULL_TRAIN.label_fileids



One of the reasons that the creation of the CIEMPIESS Corpus took so long was that 
it has "time labels" to indicate where a word begins and ends in a certain recording.

This "time labels" were created with the software PRAAT

		www.praat.org

PRAAT generates these "time labels" with the extension ".TextGrid", and that is why we 
call "textgrigs" to them.

The "time labels" or "textgrids" are only available for the 16717 "primitive" audio files.

NOTICE THAT:

The file CIEMPIESS_FULL_TRAIN.label_transcriptions was taken directly from all the
"time labels" with the help of a python script. It means that this file is a reflect
of the "time labels".

Nevertheless, the file CIEMPIESS_FULL_TRAIN.transcription was, at the beginning, equal
to the CIEMPIESS_FULL_TRAIN.label_transcriptions, but we corrected spelling errors in the
CIEMPIESS_FULL_TRAIN.transcription while the other file remained untouched.

So, in conclusion, these two files are not exactly the same but they are very similar to
each other.

-----------------------------------------------------------------------------------------------
"sphinx_experiments" DIRECTORY
-----------------------------------------------------------------------------------------------

You will find the following directories:

	- T22_NOTONIC

	- T22_TONIC

	- T50_NOTONIC

	- T50_TONIC
	

All of these directories contain the files needed to perform training and 
recognition experiments with the SPHINX3 recognition software.

The directories with the word "TONIC" have SPHINX3 files that take into account
the tonic vowel of every word, and the directories with the words "NOTONIC"
work with all the words in lowercase.

For more information about how to deal with tonic vowels you can see this article:

Carlos. D. Hernández-Mena and José. A. Herrera-Camacho, 
“CIEMPIESS: A new open-sourced mexican spanish radio corpus,” 
in Proc. LREC. European Language Resources Association, 2014. 

That you can download from here

http://www.lrec-conf.org/proceedings/lrec2014/pdf/182_Paper.pdf


Anyway, every directory have the following files:

feat.params : Contains several variables to calculate the MFCC with SPHINX3

.dic : Pronouncing dictionary

.filler : Filler dictionary

.phone : list of phonemes

.ug.lm : ASCII version of the Language Model in ARPA Format

.ug.lm.DMP : Binary version of the Language Model in ARPA Format

_test.fileids : List of all the paths to the audio files of the test set

_test.transcription : Transcription file of the test set

_train.fileids : List of all the paths to the audio files of the train set

_train.transcription : Transcription file of the train set


T22 directories handle only phonemes of the Mexican Spanish.
T50 directories handle phonemes and allophones.


NOTICE THAT:

In the directories with the word "TONIC" you will find words with double letters like these:

AAbre
bloquEEen
enIIgmas
OObras
ajUUsco
agroeKKSSportadOOr
mEEJJico
SSicotEEncatl
precampAANNa

This is because SPHINX3 and the CMU Statistical Language Modeling Toolkit 
(http://www.speech.cs.cmu.edu/SLM/toolkit.html) does not distinguish between
lowercase and uppercase. This represents a problem to the CIEMPIESS because
it utilizes uppercase letters to indicate things (for example: tonic vowels).

To handle these double letters you can do a simple "Find and Replace" and
do the following substitutions:


AA -> A
EE -> E
II -> I
OO -> O
UU -> U
KKSS -> KS
JJ -> J
SS - SS
NN -> N

-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
