# AIDL2024-Solar-Panels

##### Table of Contents  
- [Introduction and motivation](#introduction-and-motivation)  
- [The proposal](#the-proposal)  
- [Milestones](#milestones)  
- [Dataset](#dataset)  
- [Models used](#models-used)  
    - [UNet](#unet)  
    - [SegFormaer](#segformer)  
    - [Model's results comparision](#models-results-comparision)  
- [Solar panel power estimation](#solar-panel-power-estimation)
- [Next steps](#next-steps)
- [How-to run the code](#how-to-run-the-code)  
    - [How-to launch train \& validation process](#how-to-launch-train--validation-process)  
    - [How-to run inference](#how-to-run-inference)  
    - [How-to run generalization](#how-to-run-generalization)  

## Introduction and motivation

TODO: get from first slides presented to Amanda.

## The proposal

TODO: Describe your solution in terms of neural architecture , data & computational requirements.

## Milestones

TODO: Report on the degree of achievement of the milestones defined during the critical review. Stages proposed in the initial slides.

## Dataset

TODO: Information about datasets used (size, spatial-resolution, source, etc.)

## Models used

TODO: Explain why we have used two models.

### Metrics

TODO: Present F1 (Dice) and Jaccard index.

### Unet

#### Architecture

TODO: Specify the architecture used and relevant details.

#### Experiments

| # | Hypothesis | Setup | Result | Conclusion |
| --- | --- | --- | --- | --- |
| 0 | Baseline | Epochs = 10<br> Batch = 8<br>LR = 1,000E-04 | Avg. Loss = xxx<br>Avg.Accuracy = yyy<br>Dice = xxx<br>Jaccard = xxxx | The objective of this first round was to validate the model, and verify the initial behavior, verifying that the code was correct. |
| 1 | Fine-tunning LR | Epochs = 10<br> Batch = 8<br>LR = 1,000E-04 | Avg. Loss = xxx<br>Avg.Accuracy = yyy<br>Dice = xxx<br>Jaccard = xxxx | bla |

### SegFormer

#### Architecture

TODO: Specify the architecture used and relevant details.

#### Fine-tunning

TODO: Explain strategies used

#### Experiments

| # | Hypothesis | Setup | Result | Conclusion |
| --- | --- | --- | --- | --- |
| 0 | Baseline | bla | Avg. Loss = xxx<br>Avg.Accuracy = yyy<br>Dice = xxx<br>Jaccard = xxxx | bla |
| 1 | bla | bla | Avg. Loss = xxx<br>Avg.Accuracy = yyy<br>Dice = xxx<br>Jaccard = xxxx | bla |

### Model's results comparision

TODO: comparation between models and conclusions.

#### What didn't work?

#### What did work?

## Solar panel power estimation

TODO: explain which is the estiamation of the solar panel power of an area of interest generalizing with the model

TODO: add some results.

## Next steps

## How-to run the code

### Dataset preparation

#### Download and folder layout

#### Image modifications

#### Mean and std-deviation computation

### How-to launch train & validation process

### How-to run inference

### How-to run generalization
