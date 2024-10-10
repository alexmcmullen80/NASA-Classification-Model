# NASA-Classification-Model
<pre>
  The sun's solar flares are large bursts of electromagnetic energy that can significantly impact life on Earth. 
  Powerful solar flares can knock out power grids, satellite systems, and all communication systems on the planet. 
  To monitor solar activity and study such events, the United States National Oceanic and Atmospheric Administration (NOAA) 
  operates the Geostationary Operational Environmental Satellite System (GOES); a series of geosynchronous satellites with 
  specialized measurement instruments onboard. The GOES system provides us with solar imagery, magnetometer data, solar X-ray data, 
  and data on high-energy solar protons that hit Earth. This data is sent to Earth and regularly updated on the GEOS server which is 
  open for public use.
  Unlike Earth, the sun’s regions are not divided by countries, states, or cities. Instead, various patches on the sun are numbered 
  by NOAA for scientific investigation purposes, and the most active patches are frequently monitored for high-intensity bursts of 
  electromagnetic radiation. Each patch is called a HARP (HMI Active Region Patch); an enduring, coherent magnetic structure that 
  produces an electromagnetic field. The regions provide measurable features that characterize that patch. There are two classes 
  of solar flare events of particular interest to scientists: the M-class and the X-class. 
  These solar flares occur in various HARP regions, and the level of energy bursts is measured on a scale.
  While monitoring these patches for solar flares is helpful, it is much more useful if we can predict the next powerful burst. 
  Predicting an upcoming solar flare 24 hours in advance can give us a little time to prevent major disasters. 
  However, this is very challenging because solar flares are rare events. Most importantly, we do not know which features directly
  indicate an upcoming solar flare.
  
  This project consists of a ML-based binary classification model using data from the Helioseismic and Magnetic Imager Instrument 
  on NASA's Solar Dynamics Observatory (SDO) that captures various solar events.
  The model uses best combination of this data to predict the occurrence of a major solar event in the next 24 hours.
  It achieves average accuracy of around 88% which is comparable to the results of the paper published by Bobra and Couvidat
</pre>
