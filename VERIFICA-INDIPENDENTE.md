# Verifica indipendente dello stack AI con 2× Tesla P40

**Oggetto:** verifica critica di `report-verifica-stack-ai-p40.md`  
**Scenario temporale:** dati e documentazione consultati fino a fine 2026, ove disponibili  
**Obiettivo:** valutare se una build locale con due Tesla P40 sia la scelta più sensata per inferenza di modelli densi da circa 70–72B, con uso previsto di circa 16 ore al giorno e budget complessivo indicativo di 850–1.250 €.

> **Nota metodologica.** Le specifiche hardware e i requisiti software sono separati dai benchmark empirici e dai prezzi. Un benchmark Reddit/GitHub non è una prova riproducibile al livello di una specifica del produttore; un prezzo di marketplace è un annuncio o una tariffa osservata, non un listino garantito. Le cifre di prestazione senza modello, quantizzazione, contesto, backend, versione e topologia PCIe non devono essere trattate come promesse.

## 1. Sintesi esecutiva e verdetto

Il report originale coglie la motivazione principale della soluzione: **due P40 sono un modo relativamente economico per ottenere circa 48 GB di VRAM nominale**. Tuttavia, non sono la “soluzione corretta e validata” in senso generale. Sono una soluzione di compromesso che ha senso soltanto quando il costo d’acquisto iniziale è la priorità assoluta e l’utente accetta una macchina datacenter del 2016, passiva, energivora, con supporto software in fase di uscita e generazione lenta.

### Verdetto sintetico

- **Fattibilità:** sì, con llama.cpp e una quantizzazione che lasci margine; non con qualunque GGUF da 70B.
- **Qwen2.5-72B:** IQ4_XS o Q4_0 sono plausibili; Q4_K_M da 47,42 GB è troppo vicino ai 48 GB nominali per un deployment affidabile con cache e runtime.
- **Llama 3.1 70B:** è più favorevole: Q4_K_M è circa 42,52 GB e IQ4_XS circa 37,90 GB nel repository verificato, ma serve comunque spazio per KV cache e buffer.
- **Prestazioni:** 3–6 tok/s per la generazione su dual-P40 è una previsione prudente; esistono report pubblici nell’intervallo 3–4 tok/s e circa 3,3 tok/s su 70B, ma non è stato trovato un benchmark moderno, controllato e omogeneo che giustifichi 10–12 tok/s come previsione di progetto.
- **Consumi:** 550–750 W alla presa durante carico continuativo è una stima plausibile, ma è una stima da misurare; i soli TDP GPU ammontano a 500 W.
- **CUDA nel 2026:** Pascal non è “morta” a livello di driver: il ramo datacenter R580 elenca ancora P40 e supporta applicazioni CUDA 13.x. È però uscita dai percorsi moderni più comodi: CUDA 13 rimuove il supporto di compilazione per architetture precedenti a Turing e vLLM richiede compute capability almeno 7.5. Il progetto deve quindi essere vincolato a un toolchain compatibile e a llama.cpp compilato localmente.
- **Economia:** a 16 h/giorno il costo dell’elettricità può essere 78–126 €/mese con 650–750 W medi e 0,25–0,35 €/kWh. Il pareggio contro un cloud a 0,50 €/GPU·h è possibile, ma non dimostra che la P40 sia migliore: il cloud può offrire una GPU molto più veloce, avviabile solo quando serve.
- **Alternativa non-legacy concreta:** due RTX 3090 usate sono l’unica opzione consumer realmente comparabile nel budget quando la piattaforma esiste già: 48 GB nominali, Ampere/CC 8.6 e molta più banda, a fronte di circa 700 W di TDP GPU. Se il budget comprende l’intera macchina, la scelta più equilibrata è invece una RTX 3090 singola per i modelli quotidiani più piccoli e cloud per il 70B.
- **Alternativa preferita per 70B sempre locale:** se il budget lo consente, una singola GPU Ampere da 48 GB (A40 o RTX A6000) è architetturalmente migliore; nel budget 850–1.250 €, una build con una sola RTX 3090 è più veloce ma non esegue autonomamente un 70B Q4 in 24 GB. Per 70B sempre acceso, la scelta razionale è spesso **cloud on-demand/interruptible**, una dual-3090 se la piattaforma è già disponibile, oppure una 48 GB acquistata a prezzo realmente conveniente.
- **Mini PC validati criticamente:** per eliminare gran parte del rumore e del calore, un Mac Studio M2 Max 64 GB usato è la soluzione più pulita da zero, ma il throughput 70B di 8–12 tok/s non è dimostrato sul modello specifico. Un UM780 XTX con 96 GB e eGPU OCuLink è un prototipo x86 interessante, non una promessa da 5–8 tok/s; un Mini PC CPU-only resta secondario perché ricade circa nella fascia 2,5–4 tok/s.

### Raccomandazione finale

**Non acquistare due P40 alla cieca.** Acquistare una dual-P40 è ragionevole solo se:

1. il paio di schede testate costa davvero circa 300–450 € complessivi, con reso o prova;
2. il raffreddamento passivo è già risolto;
3. l’utente accetta circa 3–4 tok/s come risultato normale, non 10–12;
4. il modello principale è Qwen IQ4_XS/Q4_0 o Llama IQ4_XS/Q4_K_M con contesto contenuto;
5. il valore di avere un server locale sempre disponibile supera circa 100 € mensili di elettricità e la complessità di manutenzione.

Se il prezzo di una P40 sale oltre circa 250–300 € l’una, oppure se si richiedono bassa latenza, più utenti, contesto lungo o aggiornamenti software semplici, **non sceglierei la dual-P40**.

---

## 2. Verifica del modello e della memoria

### 2.1 Dimensioni GGUF verificate

Il repository bartowski per Qwen2.5-72B-Instruct indica queste dimensioni:

| File | Dimensione indicata | Spazio residuo su 48 GB nominali |
|---|---:|---:|
| Q4_K_M | 47,42 GB | circa 0,58 GB |
| Q4_0 | 41,38 GB | circa 6,62 GB |
| IQ4_XS | 39,71 GB | circa 8,29 GB |
| Q3_K_XL | 40,60 GB | circa 7,40 GB |
| Q3_K_M | 37,70 GB | circa 10,30 GB |
| IQ3_M | 35,50 GB | circa 12,50 GB |

Fonte primaria del file e delle dimensioni: [bartowski/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF).

Il report è corretto sui numeri principali Qwen. La correzione importante è interpretativa: **47,42 GB non è “quasi 48 GB ma quindi eseguibile”**. Le due GPU non espongono una memoria unica da 48 GB. Ogni dispositivo deve mantenere parte dei propri buffer; inoltre servono memoria per KV cache, workspace, allocator, contesto e server. La guida del repository stesso raccomanda di scegliere un file di circa 1–2 GB più piccolo della VRAM totale quando si vuole evitare di saturare la memoria.

Per Qwen2.5-72B, IQ4_XS è quindi una scelta più equilibrata di Q4_0 se il backend CUDA supporta correttamente gli I-quant. Q4_0 è un formato legacy: la sua dimensione più grande non implica automaticamente qualità migliore rispetto a un IQ4_XS calibrato con imatrix.

La configurazione ufficiale Qwen conferma inoltre che non si tratta di un modello “semplice” dal punto di vista del contesto: il modello ha 80 layer, hidden size 8192, 64 attention heads e 8 key/value heads. [config.json ufficiale Qwen](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json). La KV cache è ridotta dal rapporto GQA, ma cresce comunque con il contesto e con il numero di sequenze concorrenti.

### 2.2 Llama 3.1 70B

Il README del repository GGUF verificato indica:

| File | Dimensione indicata | Valutazione su 2×P40 |
|---|---:|---|
| Q4_K_M | 42,52 GB | plausibile con contesto contenuto |
| Q4_K_S | 40,35 GB | più margine, qualità leggermente inferiore |
| IQ4_XS | 37,90 GB | buon margine |
| Q3_K_XL | 38,06 GB | più margine, qualità inferiore |
| IQ3_M | 31,94 GB | molto margine, qualità più bassa |

Fonte: [README di Meta-Llama-3.1-70B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/raw/main/README.md).

Qui il report originale è stato troppo generico quando parla di “42–43 GB”: **Q4_K_M da 42,52 GB è una cifra verificabile**, mentre Q4_K_L è 43,30 GB. Anche il file specifico, non soltanto il nome della quantizzazione, va controllato.

Llama Q4_K_M è più realistico su due P40 rispetto a Qwen Q4_K_M, ma non significa che si possa impostare senza verifiche un contesto grande, più slot o una cache KV in F16. In un server 16 ore al giorno, partirei da un solo slot, contesto 4k e cache KV quantizzata solo in modalità `layer`, dopo aver verificato che la release utilizzata supporti quella combinazione.

### 2.3 Qualità della quantizzazione

Le quantizzazioni IQ del repository sono associate a un’imatrix; llama.cpp documenta che l’importance matrix può migliorare la qualità del risultato della quantizzazione riducendo la perdita di accuratezza. Fonti:

- [llama.cpp: imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [llama.cpp: quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

Questo non rende IQ4_XS equivalente a Q4_K_M in ogni benchmark. Significa che il confronto deve essere fatto con **perplexity e test applicativi sullo stesso modello**, non con la sola dimensione del file. L’uso di EXL2 è un’alternativa soprattutto per stack ExLlama su GPU NVIDIA moderne; non lo considererei il percorso principale su Pascal, dove la compatibilità e i kernel sono il problema prima ancora del formato.

---

## 3. Verifica hardware e piattaforma

### 3.1 Tesla P40: specifiche corrette, ma con un caveat pratico

Il datasheet NVIDIA riporta per la Tesla P40:

- 24 GB GDDR5;
- banda memoria 346 GB/s;
- PCI Express 3.0 x16;
- consumo massimo 250 W;
- formato full-height, dual-slot;
- raffreddamento passivo.

Fonte: [NVIDIA Tesla P40 Datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf).

Il datasheet è un PDF e non sempre è estraibile come testo dal browser, ma l’URL è quello ufficiale NVIDIA. Due schede portano a 48 GB nominali e 500 W di TDP massimo complessivo. Non c’è NVLink che trasformi le due memorie in un pool veloce; la comunicazione ordinaria passa dalla topologia PCIe e dal supporto P2P del sistema.

La P40 è inoltre una scheda **passiva** pensata per chassis server con un flusso d’aria canalizzato. Due schede in un case ATX con ventole generiche possono raggiungere temperature e rumorosità non accettabili. Nel budget devono comparire ventole ad alta pressione, staffe/shroud, spazio tra le schede e un test termico di alcune ore.

### 3.2 CPU e motherboard

Le specifiche Intel confermano per Xeon E5-2680 v4:

- 14 core / 28 thread;
- 2,40 GHz base e 3,30 GHz turbo;
- TDP 120 W;
- DDR4-1600/1866/2133/2400;
- quattro canali;
- banda memoria massima dichiarata 76,8 GB/s;
- 40 lane PCIe 3.0;
- AVX2.

Fonte: [Intel Xeon E5-2680 v4](https://www.intel.com/content/www/us/en/products/sku/91754/intel-xeon-processor-e52680-v4-35m-cache-2-40-ghz/specifications.html).

La MSI X99A SLI PLUS dichiara quattro slot PCIe 3.0 x16 fisici, ma la pagina MSI specifica principalmente memoria **non-ECC unbuffered** e supporto fino a 3-way. Fonte: [MSI X99A SLI PLUS – specifiche](https://www.msi.com/Motherboard/X99a-SLI-PLUS/Specification). La pagina è stata raggiunta tramite il risultato ufficiale MSI, ma può restituire 403 a seconda della regione/browser.

Il report è prudente nel dubitare dell’ECC Registered, ma la raccomandazione dovrebbe essere più netta: **non acquistare 64 GB ECC RDIMM per questa scheda come se fossero garantiti**. Il fatto che il processore Xeon supporti ECC non implica che una motherboard X99 consumer inizializzi RDIMM. Preferire 4×16 GB DDR4 UDIMM compatibili, oppure comprare RAM con diritto di reso e part number verificato.

Prima dell’acquisto vanno controllati manuale, BIOS, disposizione degli slot, Above 4G Decoding, spazio fisico e alimentazione. Le 40 lane del processore non garantiscono da sole che qualunque coppia di slot operi come x16/x16 nella configurazione reale della scheda madre.

### 3.3 Alimentatore e consumi

Il dimensionamento a 1.000 W è ragionevole come minimo pratico con un alimentatore di qualità; 1.200 W è preferibile se il sovrapprezzo è ridotto. Il dato corretto da comunicare è però il consumo misurato alla presa, non la somma dei TDP:

- TDP GPU: fino a 500 W;
- TDP CPU: 120 W;
- motherboard, RAM, SSD, ventole e perdite PSU: ulteriori consumi;
- il TDP non è uguale al consumo costante alla presa.

La stima del report di 550–750 W alla presa durante inferenza sostenuta è **plausibile come intervallo di progetto**, non un dato verificato. Va misurata con wattmetro nelle condizioni effettive. Il valore 300–350 W per l’intera macchina non è compatibile con due P40 in carico elevato, salvo un carico GPU molto lontano dal pieno utilizzo.

---

## 4. Prestazioni: cosa è supportato e cosa no

### 4.1 Evidenza disponibile

Non è stato trovato un benchmark pubblico moderno, riproducibile e controllato che confronti esattamente due P40, lo stesso GGUF Qwen2.5-72B/Llama 3.1 70B, la stessa versione di llama.cpp, lo stesso contesto e la stessa topologia PCIe. Le fonti disponibili sono quindi ancore indicative:

- la discussione LocalLLaMA riporta **3–4 token/s** per un modello 70B su due P40 con contesto 8192 e una quantizzazione della famiglia Q4; [discussione LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/);
- l’issue di llama.cpp su P40 e altra GPU cita circa **3,3 token/s** su due P40 con Miqu 70B Q4_K_M; [issue #6386](https://github.com/ggml-org/llama.cpp/issues/6386);
- una discussione del 2024 su due P40 mostra che il progetto era effettivamente usato con modelli grandi, ma l’issue è chiusa come stale e non costituisce benchmark certificato;
- una discussione recente su 3090 + P40 documenta problemi di output corrotto con una modalità di split/row sperimentale, confermando che la combinazione di GPU differenti e percorsi multi-GPU richiede test; [issue #14795](https://github.com/ggml-org/llama.cpp/issues/14795).

Questi dati sono compatibili con una previsione di **3–6 tok/s** per la generazione, con 3–4 tok/s come aspettativa più sicura per Q4/IQ4 e contesto non minimo. Il limite è soprattutto la banda memoria effettiva, la sincronizzazione tra schede e il traffico PCIe, non soltanto il numero di CUDA core.

La cifra 10–12 tok/s non è impossibile come risultato di una combinazione molto favorevole o di un carico diverso, ma **non è una base responsabile per decidere l’acquisto**. La cifra dovrebbe essere indicata come upper bound non verificato, non come previsione.

Il prompt processing può essere molto diverso dal decode: batch grandi e parallelismo possono migliorare il prefill, mentre la generazione token-per-token resta più sensibile a banda e latenza. Per un assistente interattivo la metrica principale è quindi decode tok/s, insieme a time-to-first-token, non il solo prompt processing.

### 4.2 Multi-GPU llama.cpp nel 2026

La documentazione corrente di llama.cpp distingue:

- `layer`: pipeline parallelism; ogni GPU contiene una fetta contigua di layer. È il percorso più compatibile e riduce le comunicazioni tra GPU;
- `tensor`: divide anche i tensori e la KV cache; è sperimentale, mira a ridurre la latenza ma richiede più comunicazioni e un interconnect veloce;
- `row`: indicato come deprecato/superato nella documentazione attuale.

Fonte: [llama.cpp – Using multiple GPUs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).

Per due P40 identiche partirei da:

```text
--split-mode layer
--tensor-split 1,1
--n-gpu-layers all
--ctx-size 4096
--parallel 1
```

La documentazione attuale chiarisce anche due aspetti che rendono il comando del report non universalmente valido:

1. con `--split-mode tensor`, la KV cache quantizzata non è implementata e occorre usare cache non quantizzata, oltre a Flash Attention;
2. il supporto alla modalità tensor non è disponibile per ogni architettura e il comportamento è più sperimentale.

Di conseguenza, la combinazione `--split-mode tensor` + `--cache-type-k q8_0` / `--cache-type-v q8_0` proposta nel report non va presentata come configurazione generale: **è adatta a un test solo se la specifica release lo accetta**, mentre il percorso prudente su P40 è `layer`.

La documentazione di llama.cpp indica inoltre che `GGML_CUDA_P2P=1` consente trasferimenti diretti tra GPU, ma avverte di possibili crash o output corrotti in alcune motherboard/BIOS e con IOMMU. Fonte: [llama.cpp – build e P2P CUDA](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). P2P va attivato dopo aver verificato la stabilità, non come prerequisito.

### 4.3 Speculative decoding e ruolo delle P40

llama.cpp supporta speculative decoding con un modello draft più piccolo e documenta anche modalità n-gram ed EAGLE. Fonte: [llama.cpp – speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

Non è però corretto descrivere una P40 come un “acceleratore della GPU principale” che può ricevere genericamente la KV cache o il prefill. In llama.cpp:

- la KV cache è normalmente distribuita insieme ai layer del modello;
- non esiste una VRAM unificata trasparente tra P40 e GPU moderna;
- spostare dati via CPU/RAM non è un acceleratore;
- una P40 usata per il draft può diventare il collo di bottiglia se il draft non produce token abbastanza rapidamente;
- una P40 aggiunta a una GPU moderna può rallentare una pipeline eterogenea se le assegnazioni obbligano la GPU veloce ad aspettarla.

Il pattern utile è sperimentare un piccolo draft su CPU o sulla GPU più adatta, ma non acquistare una P40 pensando che fungerà da cache esterna o coprocessore universale. Se si dispone già di una P40, si può provare una ripartizione layer e misurare; non è una giustificazione economica sufficiente per comprare la seconda.

---

## 5. Driver, CUDA, vLLM e SGLang

### 5.1 Pascal non è priva di driver, ma è fuori dal percorso moderno

La pagina NVIDIA sulle compute capability elenca ancora le architetture moderne, ma rimanda le GPU legacy a una tabella separata; la P40 è Pascal, compute capability 6.1. Fonte: [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus).

La situazione a fine 2026 va descritta distinguendo tre livelli:

1. **Driver:** il ramo NVIDIA Data Center R580 documenta esplicitamente Tesla P40 tra le GPU Pascal supportate e indica supporto alle applicazioni CUDA 13.x. Fonte: [R580.126.20 Linux release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-20/index.html).
2. **Toolkit/compiler:** le note CUDA 13 indicano la rimozione del supporto offline/compilazione per Maxwell, Pascal e Volta; CUDA 13 è quindi un driver/runtime possibile in certi casi, non un toolchain sicuro per compilare kernel `sm_61`. Fonte: [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) e [CUDA Features Archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html).
3. **Framework:** PyTorch e i framework che dipendono da wheel/kernel precompilati possono rimuovere `sm_61` prima o indipendentemente dal driver. Il progetto PyTorch ha documentato la rimozione di Maxwell/Pascal dalle build CUDA 12.8; [discussione ufficiale PyTorch](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).

Il report originale è quindi corretto nel raccomandare un ambiente fissato, ma troppo vago. “CUDA fissato” deve diventare una decisione concreta: sistema operativo, driver R580 compatibile, toolkit/compilatore che riesca a generare o eseguire il codice richiesto per `sm_61`, commit di llama.cpp e procedura di ricostruzione. Non bisogna basarsi sull’ultima immagine Docker disponibile.

### 5.2 vLLM

La documentazione attuale di vLLM richiede per NVIDIA **compute capability 7.5 o superiore**. La P40 con CC 6.1 non soddisfa il requisito. Fonte: [vLLM – GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).

Verdetto: **vLLM moderno non è una scelta supportata per la P40**. Compilare fork o vecchie versioni può essere un esperimento, ma non va indicato come stack production o a bassa manutenzione.

### 5.3 SGLang

La documentazione SGLang attuale dichiara che FlashInfer, backend di attention predefinito, supporta solo `sm75` e superiori. Anche la lista di hardware NVIDIA supportato è orientata a GPU moderne. Fonte: [SGLang – Installation](https://docs.sglang.ai/get_started/install.html).

Verdetto: **SGLang moderno non è un percorso affidabile su P40**. Si possono cercare backend alternativi e versioni storiche, ma questo aumenta il costo di manutenzione e annulla il vantaggio di un server semplice.

### 5.4 Stack raccomandato

Per la dual-P40 il candidato principale resta:

- Linux stabile;
- driver NVIDIA ancora compatibile con Pascal;
- llama.cpp compilato localmente con backend CUDA;
- architettura CUDA esplicitamente selezionata e verificata;
- test di output e stabilità su una GPU, poi su due;
- pinning del commit e conservazione dell’ambiente di build.

Il valore di questa soluzione è la capacità GGUF e la flessibilità di llama.cpp, non l’ecosistema server moderno di vLLM/SGLang.

---

## 6. Confronto delle alternative hardware

I prezzi sotto sono **fasce indicative osservate in annunci, tracker o tariffe online nel periodo consultato**, non quotazioni garantite per l’Italia. IVA, spedizione, bracket, test, reso e alimentazione possono cambiare il totale.

| Soluzione | VRAM utile nominale | Prezzo indicativo osservato | Prestazioni 70B attese | Consumi/complessità | Pro | Contro |
|---|---:|---:|---|---|---|---|
| **2× Tesla P40** | 48 GB GDDR5 | circa 300–600 € la coppia; 300–450 € è un buon acquisto | circa 3–6 tok/s; ancore pubbliche 3–4 tok/s | fino a 500 W GPU; raffreddamento passivo e multi-GPU | miglior VRAM/€ tra le opzioni economiche; 70B Q4/IQ4 possibile | Pascal, PCIe, niente NVLink, driver/toolchain legacy, rumorosa, lenta |
| **1× RTX A6000** | 48 GB GDDR6 ECC | annunci osservati spesso circa 2.600–3.800 USD/usato, talvolta molto di più | molto più veloce della P40; cifra precisa da benchmark del modello | 300 W; singola GPU, semplice | 48 GB su un solo dispositivo, Ampere, CC 8.6, supporto moderno | normalmente fuori budget; prezzo usato molto variabile |
| **1× Tesla A40** | 48 GB GDDR6 ECC | annunci da circa 1.500 USD fino a circa 2.000 € o più; varia enormemente | più veloce della P40; 48 GB singoli | 300 W, passiva, richiede airflow server | 48 GB, Ampere CC 8.6, possibile NVLink con seconda A40 | passiva e spesso fuori budget; verifica bracket e raffreddamento |
| **1× RTX 3090** | 24 GB GDDR6X | circa 550–800 € in osservazioni di mercato; può essere più alta | molto più veloce per modelli che entrano; 70B Q4 non entra tutto | 350 W; consumer, più semplice | grande banda, CUDA moderna, forte rapporto prestazioni/prezzo | 24 GB: non basta per Qwen/Llama 70B Q4 con solo VRAM |
| **2× RTX 3090** | 48 GB nominali | circa 800–1.600 € la coppia; 800–1.100 € è un’occasione locale/testata, 1.100–1.600 € una fascia più prudente; una macchina completa può arrivare a 1.300–2.000 € | circa 10–18 tok/s come ordine di grandezza, con ancore pubbliche intorno a 12,5 e 15–18 tok/s; da verificare sul modello scelto | fino a 700 W GPU; PSU 1.200–1.500 W, calore e 2 GPU | migliore alternativa consumer non-legacy; banda molto superiore, Ampere/CC 8.6, ecosistema CUDA attuale | VRAM ancora distribuita, prezzo e garanzia dell’usato variabili, spazio/raffreddamento e consumi elevati |
| **1× RTX 4090** | 24 GB GDDR6X | tipicamente oltre una 3090 usata; cifra da verificare sul mercato locale | eccellente per modelli fino a 24 GB; non risolve da sola il 70B Q4 | 450 W; supporto moderno | altissima velocità | VRAM insufficiente per il target senza offload o seconda GPU |
| **RX 7900 XTX** | 24 GB GDDR6 | circa 600–900 USD osservati su tracker/annunci | molto veloce per modelli che entrano; 70B richiede split/offload | 355 W; ROCm/Vulkan da scegliere | 24 GB, 960 GB/s, ottimo throughput memoria | non esegue 70B Q4 interamente; ecosistema meno uniforme; IQ GGUF non compatibile con Vulkan secondo il repository quantizzato |
| **2× Tesla P100 16 GB** | 32 GB HBM2 | spesso 80–200 USD l’una in annunci; molto variabile | non adatta a 70B Q4 completo; utile per modelli più piccoli | 250 W PCIe o fino a 300 W secondo variante; passiva | HBM2 a 732 GB/s nella variante 16 GB | 32 GB totali ancora insufficienti per il target; Pascal; due schede senza vantaggio sufficiente |
| **CPU-only Xeon esistente** | RAM di sistema | costo incrementale quasi nullo se hardware già disponibile | circa 2–3 tok/s come ordine indicativo pubblico, ma molto dipendente dalla RAM/CPU | consumi inferiori alla dual-P40 se la GPU è spenta; semplice | zero acquisto GPU, nessun problema CUDA | lento; banda RAM e latenza limitano il decode; 64 GB appena sufficienti per alcuni file |
| **Mac Studio M5 Max 128 GB** | memoria unificata 128 GB | prezzo Apple da verificare nel configuratore locale | benchmark indipendente da eseguire; architettura molto più adatta del dual-P40 per semplicità/efficienza | massimo continuo dichiarato 480 W; molto semplice e silenzioso | memoria unificata, Metal/MLX, niente split PCIe | costo iniziale elevato; memoria non espandibile; compatibilità GGUF/MLX da verificare per modello |
| **Mac Studio M5 Ultra 256/512 GB** | memoria unificata fino a 512 GB | Apple annuncia partenza USA da 5.499 USD per M5 Ultra | può eseguire modelli molto più grandi; prestazione da misurare sul framework scelto | massimo continuo dichiarato 480 W; semplice | enorme memoria unificata, banda dichiarata 1,2 TB/s | totalmente fuori budget; non è una soluzione low-cost |

### Fonti delle specifiche della tabella

- P40: [NVIDIA P40 Datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf).
- RTX A6000: [NVIDIA RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/).
- A40: [NVIDIA A40](https://www.nvidia.com/en-us/data-center/a40/) e [datasheet A40](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf).
- RTX 3090: [NVIDIA GeForce RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/).
- RTX 4090: [NVIDIA GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- RX 7900 XTX: [AMD Radeon RX 7900 XTX](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html).
- P100: [NVIDIA Tesla P100 Datasheet](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf).
- Mac Studio: [Apple Mac Studio technical specifications](https://www.apple.com/mac-studio/specs/) e [annuncio M5](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/).

### Lettura critica delle alternative

#### A40 e RTX A6000: la vera alternativa architetturale

Una GPU singola da 48 GB elimina la parte più fastidiosa del problema: non occorre dividere i pesi tra due dispositivi, non si dipende dal P2P PCIe per ogni configurazione, e Ampere ha supporto software molto più attuale. L’A40 ha memoria GDDR6 ECC e 696 GB/s; la RTX A6000 ha 48 GB GDDR6 ECC e 768 GB/s. Le specifiche ufficiali sono quindi nettamente superiori alla banda GDDR5 della P40.

Il limite è il prezzo. Le fasce online osservate per A6000/A40 sono incoerenti tra venditori professionali, eBay, importazione e marketplace; non bisogna trasformare un’offerta da 1.500 USD per una A40 in un prezzo medio europeo. Vanno aggiunti spedizione, IVA, bracket e una soluzione di airflow per la A40 passiva.

#### RTX 3090/4090: migliore velocità, non maggiore capacità

La RTX 3090 offre 24 GB e 936 GB/s, la 4090 24 GB e circa 1 TB/s. Questo dà un vantaggio enorme per modelli da 7–32B e per ciascuna porzione del modello, ma una singola scheda non contiene Qwen2.5-72B Q4. Due 3090 raggiungono 48 GB nominali e sono **la vera alternativa non-legacy a prezzo contenuto** quando la piattaforma X99, il case e la RAM sono già disponibili. Rispetto alle P40, però, il TDP GPU raddoppia da 500 a circa 700 W e servono alimentazione, spazio e airflow adeguati.

Le fonti di mercato sono molto disomogenee: sono stati osservati annunci italiani occasionali attorno a 400–550 € per una scheda, mentre marketplace e venditori con reso/garanzia mostrano spesso prezzi molto più alti, anche oltre 1.000 €. Perciò la fascia corretta da usare nel progetto è **800–1.600 € per la coppia**, non un prezzo unico. 800–1.100 € è un risultato favorevole da cercare localmente con test; 1.100–1.600 € è più prudente per venditori affidabili. Il costo complessivo di una macchina acquistata da zero può quindi salire a circa 1.300–2.000 €.

Le prestazioni pubbliche non sono omogenee, ma forniscono un ordine di grandezza utile: un test su Llama 3.3 70B riporta circa 12,55 tok/s su 2×RTX 3090, mentre una issue di llama.cpp cita una regressione da circa 18 a 15 tok/s su modelli 70B. Questi numeri non sostituiscono un benchmark del Qwen/Llama esatto con i parametri finali, ma rendono plausibile un intervallo di progetto **10–18 tok/s**, nettamente superiore ai 3–6 tok/s della dual-P40.

La dual-3090 non rende però la memoria unificata: Qwen2.5-72B Q4_K_M resta troppo vicino ai 48 GB nominali, proprio come sulla dual-P40. La scelta pratica è Llama 3.1 Q4_K_M o IQ4_XS e Qwen IQ4_XS/Q4_0, con contesto e slot iniziali contenuti.

La 4090 non è una soluzione “70B” per il solo fatto di essere molto veloce: il collo di bottiglia è la capacità, non solo il throughput. Una 4090 singola è ottima per modelli fino a circa 24 GB; per il 70B serve una seconda GPU o offload.

##### Configurazione dual-3090 realmente praticabile

Se la piattaforma X99 è già disponibile, la proposta concreta è:

- 2× RTX 3090 usate, ciascuna con 24 GB GDDR6X;
- Xeon E5-2680 v4 e almeno 64 GB di RAM, preferibilmente 128 GB se si prevede offload;
- alimentatore di qualità da 1.200–1.500 W, con connettori separati e cavi adeguati;
- case grande con almeno tre slot effettivamente liberi e airflow diretto sulle schede;
- llama.cpp in `layer` come baseline; `tensor` solo dopo avere verificato NCCL, Flash Attention, memoria KV e qualità dell’output.

Con una piattaforma già posseduta, 850–1.300 € aggiuntivi possono essere plausibili. Da zero, invece, non è corretto chiamarla una build da 850 €: il totale realistico è più vicino a 1.300–2.000 €, soprattutto se le schede vengono comprate con reso e se serve un PSU nuovo. Il vantaggio economico rispetto a due P40 si misura quindi in **prestazioni e supporto**, non necessariamente nella bolletta: due 3090 possono consumare più della dual-P40.

Per un budget totale rigido di 850–1.250 € la raccomandazione cambia: comprare una sola RTX 3090 usata, usarla per modelli 14–32B veloci e demandare il 70B a cloud quando serve. La seconda 3090 va aggiunta soltanto se il carico 70B è realmente continuativo e il costo dell’energia è accettabile.

#### RX 7900 XTX

AMD dichiara 24 GB e fino a 960 GB/s. Llama.cpp può essere usato con Vulkan o ROCm, ma il percorso non è equivalente a CUDA. Il repository Qwen segnala che gli I-quant non sono compatibili con il backend Vulkan; per una dual-P40 con IQ4_XS questa è una differenza pratica importante. Una sola 7900 XTX non contiene il modello Q4; due schede risolverebbero la capacità nominale ma aumenterebbero complessità e richiederebbero una valutazione separata di ROCm/Vulkan.

#### P100

Due P100 da 16 GB offrono 32 GB nominali, insufficienti per Qwen/Llama 70B Q4 senza offload sostanziale. La banda HBM2 è interessante, ma non compensa la capacità mancante. Sono una scelta da modelli più piccoli o da calcolo legacy, non una sostituzione credibile della dual-P40 per il target indicato.

#### CPU-only

Se CPU, motherboard e RAM sono già disponibili, la soluzione CPU-only può avere senso per validare il modello senza spendere subito 300–600 € in GPU. Un benchmark pubblico su Qwen2.5-72B riporta circa 3 tok/s su CPU, ma non è una misura sullo Xeon E5-2680 v4 e non va generalizzata. La banda DDR4-2400 massima dichiarata dello Xeon è 76,8 GB/s, contro 692 GB/s aggregati teorici delle due P40: il decode CPU sarà normalmente più lento.

La strategia “prima CPU, poi una P40, poi la seconda” è comunque migliore dell’acquisto simultaneo di tutti i componenti: permette di verificare modello, prompt template e tolleranza alla latenza prima di impegnare il budget.

#### Mac Studio

Apple è l’alternativa più elegante quando il requisito reale è molta memoria locale con bassa complessità. Il Mac Studio M5 Max è configurabile a 128 GB; M5 Ultra arriva a 256/512 GB. Apple dichiara per M5 Max fino a 614 GB/s e per M5 Ultra 1,2 TB/s, con massimo continuo di 480 W. L’annuncio M5 Ultra indica un prezzo USA di partenza di 5.499 USD, quindi non è dentro il budget.

Un Mac Studio usato o una generazione precedente può essere interessante, ma bisogna distinguere il costo del sistema intero dal costo della sola GPU e usare benchmark MLX/llama.cpp dello stesso modello. Un risultato pubblico su MacBook/M1/M4 attorno a 7–12 tok/s per Qwen 72B è un’ancora non omogenea, non una specifica Apple; non lo uso per promettere una prestazione.

---

## 7. Cloud, energia e pareggio economico

### 7.1 Energia locale

Assumendo 650–750 W medi alla presa per 16 ore/giorno e 30 giorni:

```text
0,65–0,75 kW × 16 h × 30 = 312–360 kWh/mese
```

A 0,25–0,35 €/kWh:

```text
78–126 €/mese
```

Questa cifra non include eventuale climatizzazione aggiuntiva. In estate il calore dissipato è praticamente un carico termico aggiuntivo nella stanza. In idle o con il modello caricato ma non in generazione, il consumo può essere diverso: serve un wattmetro e un profilo di utilizzo reale.

### 7.2 Tariffe cloud osservate

Le tariffe cloud sono dinamiche e spesso espresse per GPU-ora, senza CPU, disco, tasse o storage:

- RunPod mostra una pagina prezzi aggiornata nel 2026 e pagamenti per Pods, Serverless e Clusters; la pagina di modello RTX 3090 riporta offerte da circa 0,50 USD/h, mentre quella RTX A6000 da circa 0,53 USD/h in specifici contesti. Fonti: [RunPod pricing](https://www.runpod.io/pricing), [RTX 3090](https://www.runpod.io/gpu-models/rtx-3090), [RTX A6000](https://www.runpod.io/gpu-models/rtx-a6000).
- Lambda mostra A6000 a circa 1,09 USD/h nella pagina prezzi consultata; [Lambda pricing](https://lambda.ai/pricing).
- Vast.ai dichiara prezzi live determinati da domanda e offerta, con modalità on-demand, interruptible e reserved. La pagina live ha mostrato snapshot molto bassi, per esempio circa 0,07 USD/h per RTX 3090 e 0,13 USD/h per RTX 4090, ma l’offerta concreta, l’affidabilità e il numero di GPU disponibili cambiano continuamente. Fonte: [Vast.ai pricing](https://vast.ai/pricing).

Queste cifre **non sono direttamente confrontabili**: un 3090 cloud a 0,07 USD/h non è una macchina 48 GB, un A6000 a 0,53 USD/h può non includere lo stesso livello di affidabilità, e una tariffa interruptible può interrompere il servizio. Per un server sempre disponibile bisogna includere provisioning, storage persistente, download del modello, CPU/RAM e rischio di perdita dell’istanza.

### 7.3 Calcolo di soglia

Esempio trasparente, non previsione:

- capex dual-P40 completo: 850–1.250 €;
- energia: 78–126 €/mese;
- utilizzo: 480 ore/mese;
- cloud a 0,50 USD/GPU·h, assumendo una sola GPU sufficiente: circa 240 USD/mese prima di extra;
- cloud A6000 più caro o due GPU può costare molto di più, ma offre prestazioni/capacità diverse.

Con 1.000 € di capex e 100 €/mese di energia, contro 240 €/mese di cloud GPU-only, il risparmio teorico è circa 140 €/mese e il pareggio semplice è intorno a 7 mesi. In un intervallo largo, usando 850–1.250 € e 78–126 € di energia, il pareggio può stare circa tra 5 e 11 mesi.

Questo risultato è favorevole alla macchina locale **solo se**:

- il cloud è effettivamente acceso 480 ore/mese;
- la tariffa cloud è disponibile senza costi extra rilevanti;
- la macchina locale raggiunge prestazioni utili comparabili;
- non si attribuisce un costo alla manutenzione, al rumore, al rischio hardware e alla stanza riscaldata.

Con un cloud interruptible a 0,07–0,15 USD/h, la spesa GPU può essere circa 34–72 USD/mese per 480 ore e può risultare più economica della sola elettricità locale. Con una tariffa on-demand da 1,09 USD/h, il cloud costa oltre 500 USD/mese ma può essere molte volte più rapido. Quindi l’ammortamento non è un fatto assoluto: dipende dal **prezzo per prestazione utile**, non dal prezzo per ora isolato.

Per uso 16 ore/giorno, la build locale diventa economicamente più interessante rispetto al cloud on-demand; per uso discontinuo, sviluppo o picchi, il cloud quasi sempre vince in capitale e manutenzione.

---

## 8. Pattern software e architetturali raccomandati

### Pattern A – se si acquistano le due P40

1. Comprare prima una sola P40 con reso.
2. Verificare driver, temperatura e consumo.
3. Compilare llama.cpp in un ambiente congelato per Pascal.
4. Testare un 7–14B per validare CUDA senza aspettare ore per un 70B.
5. Aggiungere la seconda P40 soltanto dopo il test.
6. Usare `--split-mode layer`, `--tensor-split 1,1`, un solo slot e contesto 4096.
7. Iniziare con Qwen IQ4_XS o Llama IQ4_XS/Q4_K_M.
8. Misurare separatamente prompt processing, generation, TTFT, temperatura e watt alla presa.
9. Provare P2P solo dopo la baseline e disabilitarlo se compaiono crash/output corrotti.
10. Aumentare a 8192 il contesto soltanto dopo aver osservato la memoria libera di entrambe le GPU.

### Pattern B – massimizzare prestazioni/€ con budget limitato

La soluzione reale non-legacy è procedere in due fasi:

1. acquistare una RTX 3090 usata con test/reso e usarla per modelli fino a 24 GB, oppure 20–32B quantizzati, ottenendo una migliore esperienza quotidiana;
2. usare cloud per il 70B finché non si trova una seconda 3090 a prezzo verificato e finché il carico non giustifica circa 700 W di TDP GPU.

Se la piattaforma è già disponibile e la coppia costa circa 800–1.100 €, completare la dual-3090 è sensato: è la migliore opzione consumer non-legacy per il target, con una previsione indicativa di 10–18 tok/s. Se invece il costo della coppia supera circa 1.200–1.400 € o bisogna comprare anche PSU, case e piattaforma, il cloud 70B più una 3090 locale è spesso finanziariamente e operativamente migliore. Questo pattern evita di pagare 500–700 W per una risposta lenta e conserva hardware moderno riutilizzabile.

### Pattern C – 70B sempre locale, se il budget cresce

Cercare una singola A40 o RTX A6000 testata, con airflow e prezzo documentato. È più semplice da amministrare della dual-P40 e consente un margine più sano per KV cache. Non acquistare a prezzo da venditore professionale senza confrontare con il cloud: l’A6000 può costare diverse volte la build P40 e il recupero dell’investimento può non avvenire mai.

### Pattern D – ibrido locale/cloud

Tenere CPU o una GPU moderna locale per modelli rapidi, embedding, classificazione e draft; usare cloud interruptible/on-demand per Qwen/Llama 70B. È probabilmente il miglior rapporto tra latenza quotidiana, costo energetico e accesso occasionale a modelli grandi. La P40 va aggiunta solo se esiste un carico 70B locale continuo e prevedibile.

### Pattern E – multi-utente

La dual-P40 a 3–6 tok/s è adatta a un utente alla volta. Aumentare `--parallel` o il numero di slot moltiplica la memoria KV e peggiora la latenza. Per più utenti, batching e serving moderno, una singola GPU Ampere/Ada o un cloud configurato con vLLM/SGLang è nettamente più sensato; quei framework non supportano Pascal in modo corrente.

---

## 9. Correzioni puntuali al report originale

| Affermazione del report | Valutazione indipendente |
|---|---|
| Qwen Q4_K_M circa 47,42 GB | **Confermata.** Fonte Hugging Face verificata. |
| Qwen Q4_0 circa 41,38 GB e IQ4_XS circa 39,71 GB | **Confermata.** La conclusione di lasciare margine è corretta. |
| Llama 70B Q4_K_M nella fascia 42–43 GB | **Sostanzialmente corretta ma imprecisa.** Il file verificato è 42,52 GB; IQ4_XS è 37,90 GB. |
| Q4_K_M Qwen “troppo vicino” alle due P40 | **Corretta e da rendere più netta.** 47,42 GB lascia meno del margine raccomandato. |
| 3–6 tok/s dual-P40 | **Plausibile e prudente.** È l’intervallo da usare nel budget. |
| 10–12 tok/s | **Non validata.** Non usarla come previsione d’acquisto. |
| 550–750 W alla presa | **Plausibile ma stimata.** Va misurata; 500 W sono già il TDP massimo delle GPU. |
| 64 GB ECC Registered su X99A SLI PLUS | **Rischioso.** La scheda dichiara non-ECC unbuffered; preferire UDIMM verificati. |
| `--split-mode layer` come default | **Corretta.** È il punto di partenza più compatibile. |
| `tensor` con KV q8 come possibile configurazione iniziale | **Da correggere.** La documentazione corrente esclude la KV quantizzata in tensor mode; usare layer per quel test. |
| vLLM/SGLang come possibili stack futuri | **Troppo ottimistico.** vLLM richiede CC 7.5; SGLang/FlashInfer richiede sm75. |
| Pascal “terminato” nel supporto CUDA | **Da precisare.** Driver R580 supporta ancora P40, ma CUDA 13 ha rimosso il supporto di compilazione per Pascal e i framework moderni abbandonano sm61. |
| Ammortamento 8–15 mesi | **Possibile, non universale.** Dipende fortemente da tariffa cloud, ore effettive, energia, guasti e confronto di prestazioni. |
| GPU P40 riutilizzabile come co-processore/cache | **Da ridimensionare.** Nessuna espansione trasparente della VRAM; una GPU lenta può diventare collo di bottiglia. |

---

## 10. Piano decisionale concreto

### Acquistare la dual-P40 se tutte queste condizioni sono vere

- prezzo delle due schede testate non oltre circa 300–450 € complessivi;
- una coppia di RTX 3090 non è reperibile a circa 800–1.100 € con condizioni di reso/test, oppure il costo energetico aggiuntivo delle 3090 non è accettabile;
- alimentatore, case, ventole e piattaforma X99 disponibili o molto economici;
- 16 ore/giorno di utilizzo reale;
- una risposta a 3–4 tok/s è accettabile;
- Qwen IQ4_XS/Q4_0 o Llama Q4_K_M è sufficiente;
- l’utente è disposto a mantenere un ambiente CUDA legacy;
- non servono vLLM/SGLang, molte sessioni o context window molto grandi.

### Valutare prima la dual-3090 se queste condizioni sono vere

- la priorità è latenza o throughput;
- la piattaforma X99 è già disponibile;
- si trova una coppia di RTX 3090 testate tra circa 800 e 1.100 €;
- si accettano fino a circa 700 W di TDP GPU e un PSU da 1.200–1.500 W.

In questo scenario la dual-3090 è la raccomandazione principale: non è economica quanto la P40, ma è una piattaforma Ampere non-legacy, più veloce e più compatibile con il software corrente.

### Non acquistare la dual-P40 se una di queste condizioni è vera

- la priorità è latenza o throughput;
- il server sarà acceso solo occasionalmente;
- non è disponibile airflow dedicato;
- il venditore non concede reso/test;
- la RAM RDIMM e la motherboard non sono state validate;
- l’utente vuole usare le ultime release Python/PyTorch/vLLM;
- il costo totale supera 1.250 € senza includere il rischio di sostituzione;
- il cloud interruptible è disponibile a un costo inferiore all’elettricità e la latenza di avvio è accettabile.

### Test di accettazione minimo

Prima di dichiarare la soluzione riuscita, registrare:

- modello GGUF e hash/file esatto;
- versione/commit di llama.cpp;
- driver e toolkit CUDA;
- split mode e tensor split;
- contesto, batch, slot e tipi KV;
- prompt processing tok/s;
- generation tok/s su almeno 1.000 token;
- VRAM libera per GPU;
- temperatura dopo 60–120 minuti;
- watt alla presa durante idle, prefill e decode;
- eventuali errori o output corrotti con e senza P2P.

Senza questo log, “funziona” significa soltanto che il processo parte, non che la build sia economicamente o operativamente valida.

---

## 11. Conclusione finale

La dual Tesla P40 è **tecnicamente possibile ma non universalmente sensata**. Il report originale è vicino alla realtà sulla capacità/prezzo, sulle limitazioni termiche e sulle prestazioni modeste; deve però essere corretto su tre fronti:

1. il supporto Pascal non è semplicemente terminato: i driver R580 la supportano ancora, ma CUDA 13 e i framework moderni hanno chiuso il percorso di compilazione/prebuilt per `sm_61`;
2. il comando multi-GPU con tensor mode e cache KV quantizzata non è una baseline generale nella documentazione corrente;
3. il confronto economico deve includere il fatto che cloud economico e GPU moderne possono essere molto più veloci, anche quando la loro tariffa oraria sembra simile.

**Decisione raccomandata:**

- se la piattaforma è già disponibile e si trovano due RTX 3090 testate a circa 800–1.100 €, scegliere la **dual-3090**: è l’alternativa non-legacy più concreta, con circa 10–18 tok/s attesi e supporto Ampere/CC 8.6;
- se il budget totale è rigidamente 850–1.250 €, comprare prima una sola RTX 3090 e usare il cloud per il 70B; completare la seconda scheda solo dopo aver verificato prezzi, PSU, temperature e carico reale;
- se l’obiettivo è soltanto il massimo di VRAM/€, il prezzo delle P40 è davvero basso e 3–4 tok/s sono sufficienti, comprare una P40 con reso, validarla e poi completare la coppia;
- se si vuole un 70B sempre locale con semplicità, scegliere una singola A40 o RTX A6000 soltanto quando reperibile a prezzo realmente conveniente; non considerare i prezzi professionali usuali compatibili con il budget;
- se il 70B serve a intermittenza, usare un’architettura ibrida con una GPU moderna locale e cloud interruptible/on-demand.

Il verdetto complessivo è quindi: **la dual-P40 resta promossa come soluzione low-cost condizionata, ma la dual-RTX 3090 è la raccomandazione principale quando il requisito è prestazioni utilizzabili e software non-legacy**. Con budget completo rigido, la scelta più equilibrata è 1× RTX 3090 + cloud; con piattaforma già disponibile e coppia 3090 a buon prezzo, la dual-3090 sostituisce concretamente la P40 come scelta da preferire. La dual-P40 rimane sensata solo se il vantaggio iniziale di prezzo è netto e il costo operativo/tecnico è accettato.

---

## 12. Validazione delle architetture Mini PC a memoria espandibile

### 12.1 Perché cambiare metrica

La riflessione aggiuntiva propone un cambio di criterio: non massimizzare i GB di VRAM acquistati al minor prezzo, ma ridurre rumore, calore, consumo e manutenzione mantenendo una velocità sufficiente per l’uso quotidiano. Questo cambio è razionale per un sistema acceso 16 ore al giorno, ma non deve essere venduto come un modo per ottenere gratuitamente la capacità di una GPU da 48 GB.

La distinzione fondamentale è questa:

- un **Mac Apple Silicon** usa memoria unificata: CPU e GPU accedono allo stesso pool, senza split PCIe tra due schede; la memoria però non è espandibile dopo l’acquisto;
- un **Mini PC x86 con eGPU** conserva la memoria del modello nella combinazione VRAM + RAM di sistema: OCuLink accelera il collegamento, ma non crea un pool unificato e il percorso CPU/GPU dipende da quanti layer sono effettivamente sulla scheda;
- un **Mini PC CPU-only** usa la RAM come memoria del modello e non soffre di driver CUDA, ma il decode è limitato soprattutto dalla banda e dalla latenza della memoria e resta nella fascia che il committente ha già giudicato poco allettante.

**Nota metodologica.** Le bande e la capacità riportate sotto sono specifiche del produttore o calcoli teorici; i consumi sono separati dai TDP e, salvo indicazione diversa, sono intervalli di progetto alla presa; i prezzi sono snapshot di negozi/marketplace e non quotazioni garantite in Italia; per i tok/s sono indicati solo ancoraggi pubblici non omogenei. Non è stato trovato un benchmark controllato che misuri nello stesso ambiente Qwen2.5-72B Q4 e tutte e tre le architetture. Le cifre di throughput Mini PC sono quindi **ipotesi da validare**, non promesse.

### 12.2 Correzioni preliminari alla riflessione

Ci sono quattro correzioni importanti prima di confrontare le opzioni:

1. **Non esiste un “Mac mini M2 Max”.** M2 Max è stato montato nel Mac Studio e in MacBook Pro; il Mac mini della generazione pertinente è M2/M2 Pro, mentre il modello piccolo recente con la banda indicata è il Mac mini M4 Pro. La configurazione corretta è quindi **Mac Studio M2 Max 64/96 GB** oppure **Mac mini M4 Pro 48/64 GB**, non Mac mini M2 Max.
2. **La memoria macOS non è una riserva fissa di 15–20 GB.** Il sistema, i driver, il desktop, il runtime e la KV cache consumano memoria variabile. È corretto lasciare margine, ma non si può sottrarre un numero universale a ogni Mac. Un Mac mini M4 Pro da 48 GB è una configurazione stretta per un GGUF da circa 42 GB; 64 GB è la scelta più prudente, non una garanzia di qualunque contesto.
3. **SER8 non è validato come Mini PC con OCuLink nativo.** La pagina ufficiale Beelink elenca USB4 e due slot M.2, ma non una porta OCuLink; discussioni della community riportano anche problemi di compatibilità. Non va quindi messo nello stesso elenco di UM780 XTX senza un adattatore/modifica verificata. Il candidato OCuLink documentato qui è UM780 XTX; MS-01 ha invece uno slot PCIe fisico interno a mezza altezza, non una porta OCuLink esterna.
4. **Il consumo Mini PC + eGPU non è automaticamente 100–150 W.** Una RTX 4060 Ti 16 GB ha una potenza di scheda intorno a 165 W; con alimentatore e Mini PC, un carico GPU sostenuto può superare nettamente 150 W alla presa. Quel valore può essere raggiunto soltanto con carico parziale, power limit o un profilo misurato specifico.

### 12.3 Architettura A — Apple Silicon con memoria unificata

#### Configurazioni corrette

**Opzione A1: Mac Studio M2 Max 64 GB, eventualmente 96 GB usato.** Apple dichiara per M2 Max una banda di memoria di 400 GB/s; il Mac Studio M2 Max è una macchina desktop, non un Mac mini. I 64 GB sono sufficienti per iniziare con Llama 3.1 70B Q4_K_M da circa 42,52 GB o con IQ4_XS, lasciando più margine di un sistema da 48 GB; per Qwen2.5-72B Q4_K_M da 47,42 GB il margine resta troppo esiguo anche con 64 GB.

**Opzione A2: Mac mini M4 Pro 48/64 GB.** Apple dichiara fino a 64 GB di memoria unificata e 273 GB/s di banda per M4 Pro, oltre a Thunderbolt 5. Il valore di 273 GB/s è quindi confermato. Il Mac mini M4 Pro da 48 GB può essere interessante per 14–32B, ma per un 70B è una configurazione di compromesso: Qwen IQ4_XS da circa 39,71 GB o Qwen Q4_0 da circa 41,38 GB lasciano poco spazio per runtime, KV e sistema; 64 GB è preferibile. Il modello Qwen Q4_K_M da 47,42 GB è da escludere come obiettivo affidabile su 48 GB.

La banda M2 Max di 400 GB/s è superiore alla banda dichiarata dell’M4 Pro di 273 GB/s, ma non basta da sola per ordinare le prestazioni: contano anche numero di core GPU, implementazione Metal/MLX, quantizzazione, contesto, build e stato della memoria. Inoltre, la banda aggregata teorica delle due P40 (692 GB/s) non è confrontabile in modo diretto con un singolo pool Apple: nella dual-P40 il modello è diviso e il traffico inter-GPU attraversa PCIe.

#### Prestazioni: cosa si può sostenere

La riflessione indica **8–12 tok/s** su M2 Max per 70B Q4. Le fonti pubbliche consultate non offrono un benchmark riproducibile esattamente su Mac Studio M2 Max, Qwen2.5-72B, stesso file GGUF e stesso contesto. Sono disponibili ancore eterogenee: un test su MacBook M4 Max 128 GB riportava circa 6,18 tok/s su Qwen2.5-72B; un benchmark pubblico su Mac M2 Ultra per Llama 70B riportava circa 15 tok/s, ma usa un chip più grande e un altro modello. Questi dati non certificano 8–12 tok/s su M2 Max.

La formulazione responsabile è quindi:

- **6–10 tok/s come ipotesi di progetto da verificare** per M2 Max 64/96 GB con backend e modello favorevoli;
- **8–12 tok/s come fascia ottimistica**, non come specifica;
- M4 Pro 48 GB: non considerarlo automaticamente una macchina 70B; il benchmark trovato di circa 3,5 tok/s riguarda un modello/formato NVFP4 e una fonte non primaria, quindi è soltanto un’ancora debole, non una previsione GGUF Q4.

A favore di A: memoria realmente condivisa, nessun P2P PCIe da configurare, rumore ridotto e supporto Metal/MLX/llama.cpp generalmente più semplice. Contro: memoria saldata, prezzo d’acquisto elevato soprattutto per le configurazioni da 64/96 GB, impossibilità di sostituire la “GPU” e necessità di controllare la compatibilità del formato e del runtime scelto.

#### Consumo e prezzo

Apple pubblica valori massimi di sistema e non una promessa di consumo costante durante il decode di Qwen 72B. Misure indipendenti su Mac mini M4/M4 Pro riportano valori nell’ordine di alcune decine di watt in carichi comuni; questo supporta un budget di **35–70 W medi alla presa** come ipotesi da misurare, non il numero fisso “30–50 W”. Per Mac Studio M2 Max il consumo può essere maggiore del Mac mini e va misurato sulla configurazione specifica.

Il costo di 1.200–1.600 € può essere plausibile per un Mac Studio M2 Max usato/ricondizionato o per un’offerta particolare, ma non è una fascia garantita: le configurazioni da 64/96 GB hanno premi elevati e i prezzi osservati oltreoceano non sono prezzi italiani comprensivi di IVA e garanzia. Un Mac mini M4 Pro 48/64 GB nuovo può entrare in una fascia simile solo in funzione di memoria, SSD, promozioni e mercato locale; il prezzo ufficiale va verificato al momento dell’acquisto.

**Verdetto A:** è l’opzione più pulita per chi vuole 70B locale, silenzio e bassa manutenzione, ma non è la più economica in assoluto. Tra i due target, preferirei **Mac Studio M2 Max 64 GB usato con test** a Mac mini M4 Pro 48 GB se il 70B è un requisito reale. Non comprerei il Mac mini 48 GB contando su un fit “a filo”.

### 12.4 Architettura B — Mini PC x86 + eGPU OCuLink

#### Collegamento e modelli di Mini PC

OCuLink PCIe 4.0 x4 offre 64 GT/s lordi, equivalenti a circa **7,88 GB/s teorici per direzione** dopo la codifica PCIe; il throughput utile è inferiore e dipende dal controller, dal cavo e dal dock. Thunderbolt 4 dichiara 40 Gb/s complessivi, ma l’uso eGPU è ulteriormente limitato dal tunneling PCIe e dall’overhead. Dire che OCuLink è “circa 3× Thunderbolt 4” è una scorciatoia non universale: come ordine di grandezza pratico può essere circa 2–3× rispetto a una connessione TB4 limitata, ma va misurato. Non è tre volte una PCIe x16 interna.

- **Minisforum UM780 XTX:** il candidato più coerente con questa architettura. Ryzen 7 7840HS, due SO-DIMM DDR5, supporto dichiarato/riportato fino a 96 GB con 2×48 GB e porta OCuLink. È però una piattaforma di generazione 2023: BIOS, compatibilità del dock e disponibilità del modello usato vanno verificati.
- **Minisforum MS-01:** il produttore dichiara fino a 96 GB DDR5-5200 e uno slot PCIe 4.0 fisico a mezza altezza, che opera fino a PCIe 4.0 x8 secondo la pagina prodotto. È interessante per una scheda interna compatibile con ingombro e alimentazione, ma non è la stessa cosa di un’uscita OCuLink verso un dock esterno. Una soluzione eGPU richiederebbe un adattamento meccanico/elettrico e non va prezzata come plug-and-play.
- **Beelink SER8:** la pagina ufficiale dichiara Ryzen 7 8845HS, RAM DDR5 espandibile e due M.2 PCIe 4.0, USB4 e circa 65 W di TDP del sistema; non dichiara OCuLink nativo. Va escluso dalla lista “OCuLink pronto” salvo acquisto di un adattatore su M.2 e prova di compatibilità. In quel caso si perde uno slot NVMe e si aggiunge un punto di guasto.

La banda dual-channel DDR5-5600 teorica è circa **89,6 GB/s** (2 × 44,8 GB/s per canale a 64 bit); con latenze, controller e accessi reali il throughput utile è inferiore. La riflessione indicava 80–100 GB/s: è un intervallo teorico ragionevole per il sottosistema, non una velocità garantita del decode.

#### eGPU e memoria del modello

Una RTX 4060 Ti 16 GB è Ampere? No: è Ada Lovelace, compute capability 8.9, con 16 GB GDDR6 e circa 288 GB/s di banda; la potenza della scheda è circa 165 W. È una GPU moderna e molto più supportabile di Pascal, ma 16 GB contengono solo una parte di un 70B Q4. Una RTX 3060 12 GB è Ampere, circa 360 GB/s e 170 W, ma ha meno VRAM e meno potenza; è una scelta economica per modelli più piccoli, non un acceleratore garantito per Qwen 72B.

Con `llama.cpp` si può assegnare un numero di layer alla GPU e lasciare il resto in RAM. Non esiste però una regola valida “primi 15–20 layer”: la memoria occupata per layer cambia con modello e backend e va letta dai log di caricamento. Inoltre, ogni token attraversa sia la parte GPU sia la parte CPU; più offload non significa automaticamente più velocità se il collegamento e la CPU diventano il collo di bottiglia.

La riflessione dichiara **5–8 tok/s** per 70B Q4 con OCuLink + RTX 4060 Ti. Non è stata trovata una misura pubblica controllata su questa precisa combinazione. Un benchmark pubblico su due RTX 4060 Ti per un modello più piccolo riportava circa 23 tok/s, ma non è trasferibile a un 70B parzialmente offloaded. La fascia 5–8 deve quindi essere classificata come **ipotesi ottimistica da validare**; per il piano economico userei 3–6 tok/s finché un test sul modello e sul Mini PC non dimostra di più. Se il risultato reale è 3–4 tok/s, B non soddisfa il requisito espresso dal committente e diventa soltanto un esperimento efficiente.

La 4060 Ti può essere utile per un 70B proprio perché sposta i layer più costosi sulla GPU, ma la scelta della scheda deve essere comparata con una RTX 3090 usata: una sola 3090 offre 24 GB e 936 GB/s, e due 3090 risolvono la capacità nominale con prestazioni attese molto superiori, a prezzo di consumo e rumore. B vince solo se il vincolo principale è il form factor/consumo e si accetta il rischio di un throughput non verificato.

#### Costo e consumo

Il totale “circa 1.000 €” è possibile soltanto con un UM780 XTX usato/barebone, RAM acquistata bene, dock OCuLink economico e GPU usata. Una distinta realistica deve includere:

| Voce | Budget prudenziale da verificare |
|---|---:|
| Mini PC OCuLink usato/barebone | 350–600 € |
| 96 GB DDR5 SO-DIMM | 180–300 € |
| dock OCuLink + alimentatore/cavo | 150–300 € |
| RTX 4060 Ti 16 GB | 300–550 € |
| SSD, spedizione, adattatori e margine | 50–150 € |
| **Totale indicativo** | **1.030–1.900 €** |

Questa non è una quotazione: è una fascia di rischio che mostra perché 1.000 € non deve essere presentato come prezzo normale. Con una RTX 3060 12 GB si può ridurre il capex, ma si riducono anche VRAM e probabilmente il throughput; non è una sostituzione equivalente.

Per i consumi, **100–150 W alla presa non è il valore generale da usare** con una 4060 Ti. Come ipotesi prudenziale iniziale userei 120–220 W medi alla presa se il GPU load è parziale; con GPU vicina al limite e CPU sostenuta si può superare 220 W. Un wattmetro deve distinguere idle, caricamento, prefill e decode.

**Verdetto B:** è il miglior compromesso concettuale soltanto per chi vuole x86, Linux, RAM sostituibile e un’eGPU smontabile, accettando una catena meccanica/elettrica più complessa. Il target corretto è **UM780 XTX + 96 GB + una GPU moderna**, non l’elenco indistinto UM780/MS-01/SER8. Non lo promuoverei come soluzione certa da 5–8 tok/s senza un benchmark di accettazione.

### 12.5 Architettura C — Mini PC CPU-only ad alta memoria

Questa architettura è la più semplice dal punto di vista elettrico e acustico, ma è anche quella meno coerente con il rifiuto esplicito della fascia 3–4 tok/s. La quantità di RAM deve essere almeno 96 GB per lavorare con margine su Qwen/Llama 70B quantizzati; 64 GB può bastare per alcuni GGUF più piccoli, ma non lascia margine sano per sistema, cache e contesto.

La previsione **2,5–4 tok/s** è compatibile come ordine di grandezza con un benchmark pubblico che riportava circa 3 tok/s su Qwen2.5-72B in CPU-only, ma quel test non era su un UM780, MS-01 o ThinkCentre specifico e non dimostra il risultato sullo Xeon del report. La banda DDR5-5600 dual-channel teorica di circa 89,6 GB/s è superiore ai 76,8 GB/s dichiarati per il singolo Xeon E5-2680 v4, ma la banda non basta a prevedere i tok/s: contano architettura, cache, NUMA, quantizzazione e implementazione SIMD.

`--no-mmap` non è un acceleratore automatico. In genere llama.cpp può usare mmap e il sistema operativo gestisce il file; disabilitarlo può cambiare l’uso della RAM e il comportamento di paging, ma va fatto solo dopo una misura, non come impostazione prestazionale universale. I parametri da fissare sono thread, affinità, quantizzazione, contesto e modalità di accesso alla memoria.

Costo prudenziale: **700–1.100 €** se Mini PC, 96 GB e SSD sono acquistati separatamente; può essere inferiore con un sistema usato e RAM già disponibile. Consumo sostenuto plausibile: **30–70 W alla presa**, da misurare. A 16 ore/giorno ciò equivale a circa 14,4–33,6 kWh/mese.

**Verdetto C:** valido come server silenzioso per elaborazioni asincrone, test, embedding e modelli più piccoli; non è la scelta primaria per chat interattiva 70B se 3–4 tok/s non sono accettabili. Prima di spendere, conviene provare il GGUF e il prompt template sul sistema già disponibile o noleggiare una macchina per misurare la tolleranza reale.

### 12.6 Matrice comparativa validata

| Architettura | Fit 70B Q4 | Decode 70B indicativo | Potenza alla presa da pianificare | Costo indicativo | Confidenza del throughput | Giudizio |
|---|---|---:|---:|---:|---|---|
| **A1: Mac Studio M2 Max 64/96 GB** | Sì per file con margine; Qwen Q4_K_M 47,42 GB troppo stretto | 6–10 tok/s come ipotesi; 8–12 ottimistici | 35–100 W, misurare | 1.200–2.000 € usato/mercato variabile | Media-bassa | Migliore soluzione silenziosa chiavi in mano se il prezzo è buono |
| **A2: Mac mini M4 Pro 48 GB** | Solo quant molto compatte e contesto contenuto; non Qwen Q4_K_M | circa 3–6 tok/s da validare | 30–70 W, max ufficiale distinto dal consumo reale | circa 1.300–2.000 € secondo RAM/SSD/mercato | Bassa | Non consigliato come 70B principale a 48 GB |
| **B: UM780 XTX + 96 GB + OCuLink + 4060 Ti 16 GB** | Sì, ma con gran parte dei pesi in RAM | 3–6 tok/s prudenziali; 5–8 non validati | 120–220 W tipici da misurare | 1.030–1.900 € | Bassa | Interessante per x86/upgrade, non per throughput garantito |
| **C: Mini PC CPU-only 96 GB** | Sì per quant più piccole; contesto limita il margine | 2,5–4 tok/s come ordine indicativo | 30–70 W | 700–1.100 € | Media-bassa | Solo asincrono se 3–4 tok/s non bastano |
| **Dual-P40** | Sì con IQ4/Q4_0 e margine | 3–6 tok/s | 550–750 W | circa 850–1.250 € macchina completa | Media-bassa | Massima VRAM/€ ma pessima efficienza |
| **Dual-RTX 3090** | Sì con file e contesto scelti | 10–18 tok/s come stima pubblica non uniforme | oltre 700 W GPU, circa 800 W alla presa da misurare | 1.300–2.000 € da zero | Media | Prima scelta se piattaforma già disponibile e il throughput conta |

La matrice corregge la promessa “A/B/C = 8–12/5–8/2,5–4” trasformandola in livelli di confidenza. In particolare, A è l’unica delle tre con una ragione architetturale forte per la bassa complessità; B non è automaticamente più veloce di un Mac; C non risolve il requisito prestazionale.

### 12.7 Energia per 16 ore al giorno

Con 16 ore al giorno e 30 giorni, il fattore mensile è 480 ore. Per una tariffa di 0,25–0,35 €/kWh:

| Sistema | Potenza media assunta | Energia/mese | Costo/mese |
|---|---:|---:|---:|
| Dual-P40 | 650–750 W | 312–360 kWh | 78–126 € |
| A: Mac mini/Studio, stima da misurare | 35–100 W | 16,8–48 kWh | 4–17 € |
| B: Mini PC + eGPU, stima da misurare | 120–220 W | 57,6–105,6 kWh | 14–37 € |
| C: CPU-only | 30–70 W | 14,4–33,6 kWh | 4–12 € |

Il risparmio rispetto alla dual-P40 è quindi sostanziale, ma non sempre “oltre 70 €/mese”: per B può essere circa 41–112 €/mese, per A circa 61–122 €/mese e per C circa 66–122 €/mese, prima di considerare il carico reale e la climatizzazione. Ogni watt medio aggiuntivo vale 0,48 kWh/mese in questo profilo; un wattmetro è più utile di una stima basata sul TDP.

La dual-P40 dissipa nell’ambiente circa 0,65–0,75 kW termici durante il carico, cioè 2.200–2.560 BTU/h circa. In estate il costo/fastidio può essere superiore alla sola bolletta. Anche B dissipa il calore del proprio eGPU: “Mini PC” non significa automaticamente freddo o fanless.

### 12.8 Decisione armonizzata con la sezione 11

Il nuovo confronto non sostituisce il verdetto precedente sulla dual-3090:

- **Se la piattaforma X99 è già disponibile e la coppia di RTX 3090 testate costa 800–1.100 €, la dual-3090 resta la prima scelta per throughput.** È rumorosa e consuma molto, ma è la sola opzione tra quelle economiche che offre una previsione 70B nettamente sopra la fascia P40 e un ecosistema NVIDIA moderno.
- **Se si parte da zero e la priorità assoluta è silenzio/consumo, A è la scelta più coerente**, preferibilmente Mac Studio M2 Max 64 GB usato con test. Il prezzo deve essere confrontato con una singola RTX 3090 + cloud: il vantaggio è operativo, non necessariamente finanziario.
- **B è la scelta da esplorare se si vuole rimanere su Linux/x86 e mantenere RAM/GPU sostituibili**, ma non va acquistata per la promessa non verificata di 5–8 tok/s. Prima si compra/assembla il solo Mini PC con 96 GB e si misura CPU-only; poi si aggiunge il dock e infine la GPU con diritto di reso.
- **C resta secondaria.** È appropriata per background e sviluppo, non per l’interattività richiesta se 3–4 tok/s sono già considerati insoddisfacenti.
- **Una RTX 4060 Ti 16 GB o una RTX 3060 12 GB non sono equivalenti a una GPU da 48 GB.** L’offload in RAM è una soluzione ibrida con prestazioni dipendenti dal traffico a ogni token; non è un’espansione trasparente.

In sintesi: **da zero e senza rumore, A; da zero con Linux e upgrade path, B solo dopo un test; con piattaforma esistente e velocità prioritaria, dual-3090; C soltanto per asincrono.** Questa è una raccomandazione più solida del semplice intervallo teorico 3–12 tok/s.

### 12.9 Piano operativo e ricerca pezzi, inclusa Taobao

1. **Definire il test minimo prima del budget:** Qwen2.5-72B IQ4_XS o Llama 3.1 70B Q4_K_M, contesto 4096, un solo slot, 1.000 token di decode, e registrazione di tok/s, TTFT, RAM/VRAM, temperature e watt alla presa.
2. **Per A:** cercare Mac Studio M2 Max 64 GB con numero modello, seriale, stato SSD e diritto di reso; evitare Mac mini M4 Pro 48 GB se l’obiettivo è Qwen 72B Q4. Testare sia llama.cpp Metal sia MLX solo se il formato del modello è disponibile.
3. **Per B:** preferire UM780 XTX con porta OCuLink esplicitamente visibile e manuale/BIOS verificabile. Non comprare SER8 assumendo che USB4 equivalga a OCuLink; non comprare MS-01 per un dock esterno senza avere risolto la connessione PCIe interna. Verificare supporto a 2×48 GB DDR5, stabilità a 96 GB, SSD sacrificato dall’adattatore, alimentatore del dock e wake/boot dell’eGPU.
4. **Sequenza B a rischio ridotto:** Mini PC + 96 GB → benchmark CPU-only → dock OCuLink → test con GPU moderna → benchmark completo. La GPU deve avere reso; il dock e il cavo devono essere acquistati da un venditore che accetti sostituzione.
5. **Ricerca su Taobao:** usarla per confrontare barebone, RAM 2×48 GB, dock OCuLink/DEG1/DEG2 e GPU usate, ma non trattare il prezzo mostrato come costo finale. Annotare modello preciso, revisione BIOS, lingua del venditore, spedizione internazionale, IVA/dazi, commissioni dell’intermediario, garanzia, reso e compatibilità del connettore. Un prezzo Taobao senza costo di importazione e senza reso non è confrontabile con un’offerta europea.
6. **Per ogni annuncio Taobao chiedere:** foto dell’etichetta e della porta, CPU esatta, quantità/configurazione RAM, versione BIOS, test di memoria, test PCIe/OCuLink, temperatura, alimentatore incluso, video di avvio con eGPU e condizioni di restituzione. Evitare listing “engineering sample”, RAM saldata non dichiarata e dock privi di protezioni/alimentatore.
7. **Criterio di accettazione:** scartare B se non supera stabilmente 5 tok/s sul file scelto oppure se il costo completo supera una singola RTX 3090 + cloud senza offrire un vantaggio concreto di rumore/consumo. Scartare A se la memoria effettivamente libera non consente il modello e il contesto desiderati. Scartare C se il tempo di risposta non è accettabile per l’uso reale.
8. **Misurare almeno tre profili:** idle, caricamento/prefill e decode sostenuto per 30–60 minuti. La potenza nominale dell’alimentatore, il TDP e il consumo alla presa sono numeri diversi.

La riflessione Mini PC è quindi promossa come direzione progettuale, ma non come preventivo o benchmark già validato: l’unica opzione che soddisfa subito il requisito di bassa complessità è A, mentre B richiede un prototipo e C non supera il vincolo prestazionale dichiarato.

---

## Fonti consultate

### Modelli e quantizzazione

1. [Qwen2.5-72B-Instruct-GGUF – file, dimensioni e note sui quant](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF)
2. [Qwen2.5-72B-Instruct – config.json ufficiale](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/raw/main/config.json)
3. [Meta-Llama-3.1-70B-Instruct-GGUF – README e dimensioni](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/raw/main/README.md)
4. [llama.cpp imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
5. [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

### Hardware e piattaforma

6. [NVIDIA Tesla P40 Datasheet](https://images.nvidia.com/content/pdf/tesla/184427-Tesla-P40-Datasheet-NV-Final-Letter-Web.pdf)
7. [Intel Xeon E5-2680 v4 – specifiche](https://www.intel.com/content/www/us/en/products/sku/91754/intel-xeon-processor-e52680-v4-35m-cache-2-40-ghz/specifications.html)
8. [MSI X99A SLI PLUS – specifiche](https://www.msi.com/Motherboard/X99a-SLI-PLUS/Specification)
9. [NVIDIA RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/)
10. [NVIDIA A40](https://www.nvidia.com/en-us/data-center/a40/)
11. [NVIDIA A40 Datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)
12. [NVIDIA GeForce RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
13. [NVIDIA GeForce RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)
14. [AMD Radeon RX 7900 XTX](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html)
15. [NVIDIA Tesla P100 Datasheet](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf)
16. [Apple Mac Studio – specifiche tecniche](https://www.apple.com/mac-studio/specs/)
17. [Apple annuncia Mac Studio M5 Max/M5 Ultra](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/)

### Runtime e supporto software

18. [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
19. [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
20. [CUDA Features Archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
21. [NVIDIA Data Center Driver R580.126.20 – release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-20/index.html)
22. [PyTorch: rimozione del supporto Maxwell/Pascal nelle build CUDA 12.8/12.9](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)
23. [llama.cpp – multi-GPU](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
24. [llama.cpp – build CUDA e P2P](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
25. [llama.cpp – speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
26. [vLLM – requisiti GPU](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
27. [SGLang – installazione e nota FlashInfer sm75+](https://docs.sglang.ai/get_started/install.html)

### Benchmark e discussioni tecniche

28. [llama.cpp issue #6386 – P40 e modello 70B](https://github.com/ggml-org/llama.cpp/issues/6386)
29. [llama.cpp issue #14795 – comportamento multi-GPU con P40](https://github.com/ggml-org/llama.cpp/issues/14795)
30. [LocalLLaMA – Tesla P40 e llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/)
31. [LocalLLaMA – confronto CPU/GPU Qwen2.5](https://www.reddit.com/r/LocalLLaMA/comments/1fq883g/qwen_25_cpu_vs_gpu_comparison/)

### Prezzi cloud e mercato indicativo

32. [RunPod – pricing](https://www.runpod.io/pricing)
33. [RunPod – RTX 3090](https://www.runpod.io/gpu-models/rtx-3090)
34. [RunPod – RTX A6000](https://www.runpod.io/gpu-models/rtx-a6000)
35. [Lambda – pricing](https://lambda.ai/pricing)
36. [Vast.ai – prezzi live](https://vast.ai/pricing)
37. [Tracker/annunci Tesla P40](https://www.ebay.com/shop/tesla-p40-24gb?_nkw=tesla+p40+24gb)
38. [Tracker/annunci RTX 3090](https://www.reddit.com/r/LocalLLaMA/comments/1llms46/fyi_to_everyone_rtx_3090_prices_crashed_and_are/)
39. [Tracker/annunci RTX A6000](https://electronics.alibaba.com/product/used-a6000-gpu)
40. [Tracker/annunci Tesla A40](https://www.reddit.com/r/LocalLLaMA/comments/1ohvcwt/is_an_nvidia_a40_48gb_for_1500usd_a_bad_idea/)
41. [Tracker prezzi RX 7900 XTX](https://bestvaluegpu.com/history/new-and-used-rx-7900-xtx-price-history-and-specs/)
42. [Tracker/annunci Tesla P100](https://gpupoet.com/gpu/learn/card/nvidia-tesla-p100)
43. [Annunci RTX 3090 su Subito.it](https://www.subito.it/annunci-italia/vendita/informatica/?q=rtx+3090)
44. [Prezzi RTX 3090 usate – XDA Developers](https://www.xda-developers.com/used-rtx-3090-still-best-for-local-ai-in-value/)
45. [Benchmark 2×RTX 3090 vs M3 Max su Llama 3.3 70B](https://www.reddit.com/r/LocalLLaMA/comments/1he2v2n/speed_test_llama3370b_on_2xrtx3090_vs_m3max_64gb/)
46. [llama.cpp issue #5324 – prestazioni dual RTX 3090](https://github.com/ggml-org/llama.cpp/issues/5324)
47. [Qwen2.5 speed benchmark – documentazione Qwen](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
48. [Benchmark Qwen2.5 7B–72B su M4 Pro/M4 Max – discussione LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1gmi2em/geekerwan_benchmarked_qwen25_7b_to_72b_on_new_m4/)
49. [Qwen2.5-72B su CPU – confronto LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1fq883g/qwen_25_cpu_vs_gpu_comparison/)
50. [Qwen2.5-72B su MacBook M4 Max 128 GB – test LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1i7b3r1/i_did_a_quick_test_of_macbook_m4_max_128_gb/)

### Mini PC, Apple Silicon e collegamenti eGPU

51. [Apple Mac mini M4/M4 Pro – annuncio e specifiche di memoria/banda](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/)
52. [Apple Mac mini – consumi e output termico](https://support.apple.com/en-us/103253)
53. [Apple Mac Studio M2 Max – specifiche tecniche](https://support.apple.com/en-us/111835)
54. [Minisforum MS-01 – pagina prodotto e specifiche](https://store.minisforum.com/products/minisforum-ms-01-workstation)
55. [Minisforum UM780 XTX – pagina prodotto](https://www.minisforum.com/products/elitemini-um780-xtx)
56. [Beelink SER8 8845HS – pagina prodotto e specifiche](https://www.bee-link.com/products/beelink-ser8-8845hs)
57. [NVIDIA GeForce RTX 4060 Ti – specifiche ufficiali](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/)
58. [NVIDIA GeForce RTX 3060 – specifiche ufficiali](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)
59. [Minisforum DEG2 – dock OCuLink PCIe 4.0×4](https://minisforumpc.eu/products/minisforum-deg2-oculink-egpu-dock)
60. [OCuLink eGPU hands-on – confronto con USB4/Thunderbolt](https://www.xda-developers.com/oculink-egpu-hands-on/)
61. [Benchmark eGPU OCuLink/Thunderbolt per inferenza – eGPU.io](https://egpu.io/forums/pro-applications/impact-of-egpu-connection-speed-on-local-llm-inference-in-multi-egpu-setups/)
62. [Prezzi indicativi UM780 XTX e dock OCuLink – store Minisforum](https://store.minisforum.com/collections/oculink-compatible)

I prezzi ai punti 37–44 e 62 sono stati usati soltanto come indicatori di ordine di grandezza. Gli annunci singoli, in particolare, non provano un prezzo di vendita realizzato. Non sostituiscono una verifica contemporanea dell’offerta locale, del venditore, della garanzia, della spedizione e delle imposte.
