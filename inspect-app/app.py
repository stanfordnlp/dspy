from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
import time

app = Flask(__name__)
CORS(app)

dynamo_resource = boto3.resource("dynamodb", region_name="us-west-1")
table = dynamo_resource.Table('dsp-inspect-app')


@app.route("/")
def index():
  return "This is the main page."


@app.route("/data/<id>", methods=["GET"])
def get_item(id):
  response = table.get_item(Key={"id": id})

  if 'Item' in response:
    return jsonify(response['Item'])
  else:
    return 'Item not found', 404


@app.route("/inspect-db", methods=["GET"])
def inspect_db():
  return table.scan()


@app.route('/log-item', methods=['POST'])
def log_item():
    data = request.get_json()

    if 'id' not in data or 'content' not in data:
        return 'Missing required fields', 400

    data['expiry_time'] = int(time.time() + 86400)
    table.put_item(Item=data)

    return 'Data created successfully', 201
    

if __name__ == "__main__":
  app.run()