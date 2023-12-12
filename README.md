### YOLOV8 API FLASK

## API Reference

#### Get detection

```http
  GET /detect
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `show` | `string` | **Required**. BLOB-PARAMETERS-URL |
| `image_file` | `file` | **Required**. JPG-PNG |

#### Get image

```http
  GET /uploads/${filename}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `filename`      | `string` | **Required**. filename |



## Installation

Install dependencies
- python 3.9
- conda enviroment

```bash
  pip install -r req.txt
```
    
## Authors

- [@achilan](https://www.github.com/achilan)


## Feedback

If you have any feedback, please reach out to us at anthonychilan@icloud.com