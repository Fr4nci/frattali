# Frattali: semplice script per la generazioni

## Introduzione
Questo che ho realizzato qua è un semplice script da me ideato per generare dei frattali. Al momento bisogna intervenire manualmente per modificare il tipo di frattale ottenuto, ma comunque è tutto completamente funzionante

L'idea dietro i frattali generati è la seguenti: si considera una successione definita per ricorrenza del seguente tipo
$$z_{n+1} = z_n^2 + c$$

con $z_{n+1}, z_n, c \in \mathbb{C}$. Il comportamento dipende in particolar modo dalla costante $c$ considerata.
Per ottimizzare i tempi ho utilizzato le funzionalità di CUDA e Numpy per velocizzare i calcoli.
All'interno vi sono due script: il primo serve per generare il set di Mandelbrot (in cui $\forall z \in \mathbb{C}$ si fa variare la costante $c \in \mathbb{C}$), mentre con l'altro script è possibile andare a generare il set di Julia a $c$ costante, modificando opportunamente il codice

## Requirement
Per funzionare questo script necessita la presenza della libreria _matplotlib_, _numpy_, _pillow_, _numba_ e necessita di tutti gli strumenti installabile tramite il comando
```bash
conda install cudatoolkit
```
## Qualche immagine generata tramite gli script presi nelle cartelle
![mandelbrot_set](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/mandelbrot_set.png)
![julia_set](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/julia_set.png)
![julia_set_2](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/julia_colored.png)
![burning_ship](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/immagine_zoom_burning_ship.png)

## Nota
Per coloro che fossero interessati all'implementazione dello zoom, ho sostanzialmente utilizzanto gli _event_handler_ che la libreria matplotlib rende disponibili. Vedrete che le coordinate _y_max_ e _y_min_ nella generazione del nuovo frattale sono invertiti 
