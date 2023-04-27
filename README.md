# Inference API Server

A quick python-flask server that exposes endpoints to perform various AI inference tasks.

The goal of this server is to expose a highly configurable `/enrich` endpoint. This endpoint accepts either GET or POST headers, and acts accordingly.

- [Inference API Server](#inference-api-server)
  - [Quickstart](#quickstart)
  - [Swagger UI](#swagger-ui)
  - [Available endpoints](#available-endpoints)
    - [Examples](#examples)
  - [Options](#options)
  - [Models](#models)
    - [Creating a new model](#creating-a-new-model)
  - [GET vs. POST Requests](#get-vs-post-requests)


## Quickstart
Follow these steps to get started.

``` shell
# Start the server
flask --app src/app.py run
```

## Swagger UI

Go to the `/api` endpoint to view the swagger UI implementation. It provides working examples of how to use the endpoint. Which is likely more helpful than this doc

## Available endpoints

The primary endpoint for the api is the `/api/enrich` endpoint. This endpoint is highly configurable exposes the following options as query or POST body params (depending on method type)
- `src`: The target(s) in the form of a single string (query) or string[] (post body)
- `mode`: The type of enrichment to run (i.e., classification, OCR)
- `model`: The type of model to run. Optional configuration, only used in specific modes.

### Examples

``` js
// Perform image classifcation and return as image
POST /api/enrich
{
  "src": [
    "https://www.example.com/images/dog.jpg",
    "https://www.example.com/images/cat.jpg",
    "https://www.example.com/images/penguin.jpg",
  ],
  "format": "image",
  "model": "ResNet"
}
```

## Options

Presented below are the exposed configuration options.

<!-- Configuration Options Table -->
<table>
  <th>
    <tr>
      <td><b>Configuration Key</b></td>
      <td><b>Available Options</b></td>
      <td><b>Default</b></td>
      <td><b>Description</b></td>
    </tr>
  </th>
  <tbody>
    <!-- SRC -->
    <tr>
      <td><code>src</code></td>
      <td><code>string | string[]</code></td>
      <td><b>None (required)</b></td>
      <td>The targets (typically a remote image url)</td>
    </tr>
    <!-- Format -->
    <tr>
      <td><code>format</code></td>
      <td><code>"default" | "img" | "json"</code></td>
      <td><code>"default"</code></td>
      <td>The response output format</td>
    </tr>
    <!-- Format -->
    <tr>
      <td><code>model</code></td>
      <td><code>See models section</code></td>
      <td><code>"ResNet"</code></td>
      <td>Model to run the src through</td>
    </tr>
  </tbody>
</table>

## Models
Models are designed in a way to be highly configurable. The default models include
- `ResNet`



Note: This list is likely to change.

### Creating a new model
To create a new model, you need to create a class that implements the ModelBase class and its abstract methods. 

## GET vs. POST Requests
The `/enrich` API endpoint accepts both GET and POST headers. All configuration options remain the same however the differences are
- `src` is a **single** string param in GET as opposed to an array in POST
- Configuration keys are the query params, whereas in POST they are the POST's JSON body

It is recommended to use the GET endpoint for single requests, and the POST endpoint for batch requests.
