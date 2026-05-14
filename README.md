# CS 4210 Final Project

## Updated project direction

This project now builds on the existing TensorFlow Hub arbitrary image stylization model instead of replacing it with a brand-new model. The goal is to treat the pre-trained model as a baseline, then test ways to improve the final output so the generated image preserves style features more clearly while keeping recognizable content.

That direction matches the intermediate progress report:

- the current pipeline already loads content and style images successfully
- the TensorFlow Hub model already produces a valid stylized output
- the main weakness is that some style images are not reflected strongly enough in the result

## What changed in this repo

The original notebook is still here as the prototype:

- [style_transfer.ipynb](/Users/addison_taylor/Desktop/CS 4210/final_project/CS4210_Final_Project/style_transfer.ipynb)

A new reusable experiment pipeline has been added:

- [style_transfer_pipeline.py](/Users/addison_taylor/Desktop/CS 4210/final_project/CS4210_Final_Project/style_transfer_pipeline.py)

This new script keeps the same TF Hub model but adds controls for:

- repeated style passes to intensify style transfer
- content preservation blending to stop the output from drifting too far
- style strength adjustment to test stronger or softer stylization
- comparison set generation for report figures and analysis

## Suggested framing for the final project

You can now describe the project like this:

> This project evaluates and improves a pre-trained image style transfer model by testing output-control strategies that better preserve artistic style while retaining scene content.

That gives you a cleaner research question than "train a new model from scratch" and fits the code you already have.

## Example usage

```bash
python3 style_transfer_pipeline.py \
  --content content/content_demo_img.jpeg \
  --style style/style_demo_img.jpeg \
  --style-passes 2 \
  --content-preservation 0.15 \
  --style-strength 1.1 \
  --generate-comparison-set
```

## Good next report/demo angles

- Compare one-pass output vs multi-pass output.
- Compare low vs moderate content preservation.
- Show which settings work best for detailed vs uniform style images.
- Discuss the tradeoff between stronger style transfer and content clarity.
