# Setting up DSP function inspection

## Set up local React development environment
From `/inspect-app/react-app`, run the following commands
1. `npm install`  
2. `npm run start`

## Set up AWS DynamoDB instance
You may refer to https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/SettingUp.html for setting up a DynamoDB instance either locally or on the web service. In particular, you need to create a table with primary key named `id` and add your table name to line 10 of `/inspect-app/app.py`. You will then need to connect the flask backend to your DynamoDB instance by setting your AWS configurations appropriately and filling in line 9 of `/inspect-app/app.py`.

## Set up flask server
From `/inspect-app`, run the following commands  
1. `export FLASK-APP=app`
2. `flask run`

## Using the function inspection tool
You can toggle function inspection by creating a `FuncInspector` object and adding it as a setting. After you run your DSP function, you can use `view_data()` to generate a link to a React frontend that will display details about the function.

Setting up function inspection  
```Python
inspector = dsp.FuncInspector()
with dsp.settings.context(inspect=inspector):
  dsp_program()
```
Inspecting the states in `dsp_program`
```Python
inspector.view_data()
```
