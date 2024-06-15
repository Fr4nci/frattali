# Frattali: semplice script per la generazioni

Questo che ho realizzato qua è un semplice script da me ideato per generare dei frattali. Al momento bisogna intervenire manualmetne per modificare il tipo di frattale ottenuto, ma comunque è tutto completamente funzionante

L'idea dietro i frattali generati è la seguenti: si considera una successione definita per ricorrenza del seguente tipo
$$z_{n+1} = z_n^2 + c$$

con $z_{n+1}, z_n, c \in \mathbb{C}$. Il comportamento dipende in particolar modo dalla costante $c$ considerata.
Per ottimizzare i tempi ho utilizzato le funzionalità di CUDA e Numpy per velocizzare i calcoli
