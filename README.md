# NASA-Classification-Model
Solar Flare Prediction

## Overview

Solar flares are large bursts of electromagnetic energy emitted by the sun, capable of significantly impacting life on Earth. These events can disrupt power grids, satellite systems, and global communication networks. Predicting solar flares in advance is essential to mitigate potential disasters and maintain critical infrastructure.

## Background

The United States National Oceanic and Atmospheric Administration (NOAA) operates the Geostationary Operational Environmental Satellite System (GOES), a series of geosynchronous satellites equipped with instruments to monitor solar activity. These satellites provide:

#### Solar imagery

#### Magnetometer data

#### Solar X-ray data

#### High-energy solar proton data

The data is updated regularly and made available to the public via the GOES server. Active regions of the sun, called HARP (HMI Active Region Patch), are monitored for solar activity. These regions are identified by their coherent magnetic structures and measurable features. Scientists are particularly interested in two classes of solar flares:

#### M-class flares: Moderate-intensity flares

#### X-class flares: High-intensity flares

Predicting these solar flares is challenging due to their rarity and the lack of clear indicators from solar data.

## Project Description

This project aims to develop a machine learning (ML)-based binary classification model to predict major solar flare events 24 hours in advance. The data is sourced from the Helioseismic and Magnetic Imager Instrument on NASA's Solar Dynamics Observatory (SDO).

### Key Features

**Data Source**: Solar event data captured by the SDO.

**Model Objective**: Predict the occurrence of M-class or X-class solar flares within the next 24 hours.

**Performance**: Achieves an average accuracy of approximately 88%, comparable to results from Bobra and Couvidat's research.

## Technical Details

**Data Preprocessing**: Features are extracted from solar event datasets for training the ML model.

**Machine Learning Model**: The model evaluates the best combination of features to predict solar flare occurrences.

**Evaluation Metrics**: Accuracy and other classification metrics are used to assess model performance.

## Significance

Early prediction of solar flares provides critical lead time to:

Protect power grids from overloads.

Safeguard satellites and communication systems.

Enhance disaster preparedness for space weather events.

This work contributes to the growing field of space weather prediction and provides a foundation for further improvements in predictive models for solar activity.
