# Inference API Server

A quick python-flask server that exposes endpoints to perform various AI inference tasks

## Quickstart
Follow these steps to get started

``` shell
```

## Available endpoints

The API exposes the following public endpoints

### Image classification with JPEG response

``` shell
# Endpoint
GET /image?url=<image_url>

# Example

```

### Image classification (Group) with JPEG response

``` shell
# Endpoint
POST /combined

# Example


```

### Image classification (Raw) with JSON response

``` shell
# Endpoint
GET /image_raw?url=<image_url>

# Example
```

### Image classification (Raw Group) with JSON response

``` shell
# Endpoint
POST /combined_raw
```

### OCR detection (using tesseract) with TEXT response

``` shell
# Endpoint
GET /image_ocr?url=<image_url>

# Example
```

### OCR detection (Group, using tesseract) with TEXT response

``` shell
# Endpoint
POST /combined_ocr

# Example
```