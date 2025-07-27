# Part 1

## 1. Why did you choose your specific model and optimization technique (LoRA or Quantization)?

I chose to use **LoRA** (Low-Rank Adaptation) in combination with a **4-bit quantized** version of the `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` model because of my limited GPU RAM.

- **LoRA** allows me to fine-tune only a small set of low-rank adapter matrices instead of updating all of the model’s parameters, drastically lowering the memory required during training.
- The **4-bit quantization** reduces the model’s weight precision to just 4 bits, cutting the memory footprint nearly in half (or more) while still retaining most of the model’s representational power.

## 2. Explain the core concept behind the technique you chose. How does it reduce the computational or memory footprint of a large model?

- **LoRA (Low-Rank Adaptation)**  
  Instead of updating every weight in a massive pre-trained model during fine-tuning, LoRA freezes the original weights and injects small trainable matrices of rank _r_. During each forward pass, these low-rank matrices approximate the weight updates:

  $$
  W' = W + A B
  $$

  where $A\in\mathbb{R}^{d\times r}$ and $B\in\mathbb{R}^{r\times k}$. Because $r \ll \min(d,k)$, the number of trainable parameters (and their gradients) is dramatically smaller, reducing both GPU memory and compute requirements during back-propagation.

## 3. What are the potential trade-offs of using this technique (e.g., performance drop, accuracy changes)?

- **Accuracy Degradation**

  - LoRA’s low-rank constraint may not perfectly capture all fine-tuning nuances if the adaptation rank $r$ is set too low.

- **Performance Overhead**

  - If you set LoRA’s rank too high, you approach updating the full weight matrix, which diminishes the memory savings and may even slow down training.

# Part 2

## 1. Main Challenges

- **Limited Memory on Local Machine:** Loading the transcription models and LoRA adapters requires substantial RAM. On my local machine, available memory is insufficient for real-time inference, leading to dropped frames or high latency.
- **Platform Constraints (Google Colab):** While Colab provides ample compute, it doesn’t support live MP3 recording from a microphone or audio interface, limiting testing to pre-recorded files.
- **Real-time Processing:** Transcribing streaming audio demands optimized pipelines with low overhead to keep end-to-end latency minimal.
- **Streaming Framework Support:** Many transcription libraries focus on batch (file-based) processing and lack native streaming APIs.

## 2. Handling Continuous Audio Flow

If i were to have a good enough hardware (Ram) to do live transcription, I would do:

1. **Fixed-Length Segmentation:** Audio is segmented into short frames (e.g., 1 or 2 seconds each).
2. **Overlap & Buffering:** Apply slight overlaps (e.g., 200–300ms) between chunks to prevent cutting off words at segment boundaries.
3. **Sequential Inference:** Each chunk is immediately processed by the transcription model, emitting partial transcripts that are stitched together for a continuous result.

## 3. Production Environment Improvements

For a robust, production-ready system, I will consider these enhancements:

1. **Upgrade Hardware Infrastructure:** Deploy on servers or instances equipped with ample RAM and GPU resources to load full models and execute real-time inference without bottlenecks.
2. **Integrate True Live Audio Capture:** Build a live audio ingestion pipeline using appropriate I/O libraries (e.g., PyAudio, PortAudio, or WebRTC) to capture microphone or telephony streams directly, bypassing file-based workflows.

---

# Report on Part 1 and 2

Firstly, I read the Take-Home Test document thoroughly. This is important so that I don't take any mistep or if I have any follow up question about the test. Next, I see that I can choose between 4 models. I note that information first and then I go to the next point which is to see what we're gonna do with the model. There are 2 options: LoRA and quantization. I note this too and go to the next part of the take home test. I notice that in the second part, we need to make real time streaming transcription. Because there is a bonus when using the model that we made in part 1, I think that I should make LoRA adapter in part 1 of the take home test.

I went back to the first part and I need to decide what model do I want to make LoRA adapter. I choosed `mistralai/Voxtral-Mini-3B-2507` because I see that it's the newest model and I'd like to play with it. I know that it's the newest model because I compare the first commit of `google/gemma-3n-E2B-it` and `mistralai/Voxtral-Mini-3B-2507` and I see that `mistralai/Voxtral-Mini-3B-2507` has the initialization date that is newer than `google/gemma-3n-E2B-it`. Little did I know, this was a mistake.

Like a good engineer, I try to plan ahead on what I should do. I got the Take Home Test document on thursday night and my submission should be on monday. I decided that I would make a good LoRA adapter using bahasa Indonesia model in 3 days so that I can make the part 2 of the task in the remainding time and document my code in a good manner.

On my first 2 days, my hindarance was trying to get the `mistralai/Voxtral-Mini-3B-2507` model to accept the `mozilla-foundation/common_voice_17_0` dataset. For some reason, I have some problem with the tokenizer that doesn't work well. I am getting the problem where the `pad_token_id != eos_token_id`.

![alt text](/images/image.png)

I've been stuck and frustated on this piece of the code for hours in 2 days. I've looked up the [Mistral tokenization documentation](https://docs.mistral.ai/guides/tokenization/) and I see that there is no tokenizer for `mistralai/Voxtral-Mini-3B-2507` yet. I also assume that there is still bugs because this is still a new model. Because of this, I decided to not continue with the model and continue with `google/gemma-3n-E2B-it`.

![alt text](/images/image-2.png)

When trying the `google/gemma-3n-E2B-it` model, I always get memory error. This is because I don't have a lot of memory (note that I am using google colab). Because of this I tried to do load model with quantization. I tried this but it throws this error at me:

![alt text](/images/image-3.png)

![alt text](/images/image-1.png)

At the time, I thought that the error was caused by the fact that I am using `torch.bfloat16` so that I can represent my model in `16-bit floating point` representation. I thought the error was funny because 16 is clearly divisible by 64 and the error was nonsensical. Now that I am writing this report, after some investigation, I think that there was a problem withthe model's vocabulary size that is not a multiple of 64. I don't get it at first because at that time I have been coding for a long time (3 days of debugging).

So, I need to make a decision because the deadline is in 2 days. I decided to take a step back and see what model I can use. Long story short, I tried to use whisper but I suppose the model is too old and there was a lot of problem with the dependency so I can't use it.

I finally get the answer when I see that there is a version of quantized `google/gemma-3n-E2B-it` model by unsloth. I tried training the LoRA adapter and it worked with some modification for the `mozilla-foundation/common_voice_17_0` dataset. I used google colab for training the LoRA adapter in 2 steps. First, I used small dataset to make sure the code works first then I increase the dataset size to make the LoRA adapter have meaningful weight.

I think that the story of this first part of the take home test reveals the engineering part of me. It was the engineering decisions that I need to make so that I can achieve the goal of demonstrating to you the process of creating and saving LoRA adapter in spite of the constraints that I have (time and computational resource). I think that I have achieve the goal given the constraints.

![alt text](/images/image-4.png)

---

On the second part of the take home test, I am tasked with making a live transcription script using the `google/gemma-3n-E2B-it` base model and the LoRA adapter that I've made. This doesn't come with no challenges.

First, I tried to load the model and adapter to my local machine. Once again, my local machine doesn't have the hardware requirements to run it (see /images/image below, it's the resource needed to do inference. My local machine doesn't have enough RAM). Because of this, I once again try google colab.

![alt text](/images/image-5.png)

After a long time spent trying to code a workable live-streaming solution within Colab and the time that is ticking to the deadline, I made a decision to modify the scope of the task. It became clear that the primary goal was to demonstrate the model's transcription capabilities, not to solve a complex web engineering problem specific to Colab's limitations.

Therefore, I pivoted my approach to perform inference on a pre-recorded audio file in MP3 format. This plan allows for a clear and effective demonstration of the core requirement to prove that the base model and my custom LoRA adapter can work accurately transcribe bahasa Indonesia audio in a accaptable inference time. By using a static audio file, I can showcase the model's performance and the success of the fine-tuning process.
