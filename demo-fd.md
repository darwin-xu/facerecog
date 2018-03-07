# Client
## Detect  mode:
Detect the face of each frame. Search the face in the database. Draw a rectangle surround of the face and the name under it.
- This is the default mode.
- App needs to capture the video from CAM (in raw data).
- Data encoded into jpg and send it to server `/detectFace`.
- The result get from server will be used to draw a frame around the face and put a name under it.

## Register  mode:
Take several pictures of a man. Make sure the face is large and clear.
- A button actives the Register  mode, auto capture count down in 3 seconds.
- Put a translucent mask in the preview window.
- If there are more than one face, count down start from 3 seconds. Shows a warning messag.
- App captures 30 (TBD) pictures, send to server `/registerFace/{id}`.
- All error during the register mode will abandon the data and exit the register mode, user need to start over.

## Roll call mode:
Input the roll call name list. Check the name list frame by frame. Indicate a “PASS” sign till all the names are checked.
- A button actives the Roll call mode.
- In checkbox/textbox inputs the name list of roll call,
- App capture the video, encode and send to server `/detectFace`.
- The result get from server will be used to draw a frame around the face and put a name under it.

# Server
## POST `/detectFacesC`
### Content: jpeg data.
Detect the faces (SVM version) in the picture and return the positions and IDs of them.
### Response: json object of face list.
```json
{
   "faces":[
      {
         "possibility":0.8,
         "x1":220,
         "y1":80,
         "x2":240,
         "y2":100,
         "id":"783824f0-69a3-47a5-ad46-7b774ad75eef"
      },
      {
         "possibility":0.6,
         "x1":120,
         "y1":40,
         "x2":160,
         "y2":80,
         "id":"06430512-8217-4bfd-8946-e9da63173c71"
      }
   ]
}
```
## POST `/detectFacesD`
Detect the faces (Norm version) in the picture and return the positions and IDs of them.

## POST `/registerFace/{id}`
### param {id}: GUID to indicate the face.
### Content: jpeg data.
The picture associates to the same id will be treated as a same person.
Several pictures can be associated to the same person.
If there are more than one face on the picture, return false.
### Response: json result to indicate the result.
```json
{
   "result":true
}
```
## POST `/classifyFace`
After registered a bunch of faces, call this interface to do the classify. Otherwise the /detectFacesC wouldn't work.
### Response: json result to indicate the result.
```json
{
   "result":true
}
```

## GET `/getIds`
Get the current register faces' ID in the system.
### Response: json result of the faces' ID.
```json
{
   "result":true
}
```

## DELETE `/removeFace`
Remove all faces embedding from the system.
### Response: json result to indicate the result.
```json
{
   "result":true
}
```

## DELETE `/removeFace/{id}`
### param {id}: the GUID of the faces to remove.
Remove the faces embedding from the system.
### Response: json result to indicate the result.
```json
{
   "result":true
}
```

# Usage
1. Download the model.
2. Download the facenet.
3. Start the server: `python3 app.py`
4. Take serveral pictures of a person and put all of them in the folder.
5. Register the face: `python3 test_client/register.py <id> <folder>`
6. After all people's picture is down, do the classifying: `python3 test_client/classify.py`
7. Take another picture and test by: `python3 test_client/detect.py [-d/-c] <picture>`
