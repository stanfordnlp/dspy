# Tutorial: Use MCP tools in DSPy

MCP, standing for Model Context Protocol, is an open protocol that standardizes how applications
provide context to LLMs. Despite some development overhead, MCP offers a valuable opportunity to
share tools, resources, and prompts with other developers regardless of the technical stack you are
using. Likewise, you can use the tools built by other developers without rewriting code.

In this guide, we will walk you through how to use MCP tools in DSPy. For demonstration purposes,
we will build an airline service agent that can help users book flights and modify or cancel
existing bookings. This will rely on an MCP server with custom tools, but it should be easy to generalize
to [MCP servers built by the community](https://modelcontextprotocol.io/examples).

??? "How to run this tutorial"
    This tutorial cannot be run in hosted IPython notebooks like Google Colab or Databricks notebooks.
    To run the code, you will need to follow the guide to write code on your local device. The code
    is tested on macOS and should work the same way in Linux environments.

## Install Dependencies

Before starting, let's install the required dependencies:

```shell
pip install -U "dspy[mcp]"
```

## MCP Server Setup

Let's first set up the MCP server for the airline agent, which contains:

- A set of databases
  - User database, storing user information.
  - Flight database, storing flight information.
  - Ticket database, storing customer tickets.
- A set of tools
  - fetch_flight_info: get flight information for specific dates.
  - fetch_itinerary: get information about booked itineraries.
  - book_itinerary: book a flight on behalf of the user.
  - modify_itinerary: modify an itinerary, either through flight changes or cancellation.
  - get_user_info: get user information.
  - file_ticket: file a backlog ticket for human assistance.

In your working directory, create a file `mcp_server.py`, and paste the following content into
it:

```python
import random
import string

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Create an MCP server
mcp = FastMCP("Airline Agent")


class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}


@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights


@mcp.tool()
def fetch_itinerary(confirmation_number: str):
    """Fetch a booked itinerary information from database"""
    return itinery_database.get(confirmation_number)


@mcp.tool()
def pick_flight(flights: list[Flight]):
    """Pick up the best flight that matches users' request."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]


def generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


@mcp.tool()
def book_itinerary(flight: Flight, user_profile: UserProfile):
    """Book a flight on behalf of the user."""
    confirmation_number = generate_id()
    while confirmation_number in itinery_database:
        confirmation_number = generate_id()
    itinery_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinery_database[confirmation_number]


@mcp.tool()
def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Cancel an itinerary on behalf of the user."""
    if confirmation_number in itinery_database:
        del itinery_database[confirmation_number]
        return
    raise ValueError("Cannot find the itinerary, please check your confirmation number.")


@mcp.tool()
def get_user_info(name: str):
    """Fetch the user profile from database with given name."""
    return user_database.get(name)


@mcp.tool()
def file_ticket(user_request: str, user_profile: UserProfile):
    """File a customer support ticket if this is something the agent cannot handle."""
    ticket_id = generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id


if __name__ == "__main__":
    mcp.run()
```

Before we start the server, let's take a look at the code.

We first create a `FastMCP` instance, which is a utility that helps quickly build an MCP server:

```python
mcp = FastMCP("Airline Agent")
```

Then we define our data structures, which in a real-world application would be the database schema, e.g.:

```python
class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float
```

Following that, we initialize our database instances. In a real-world application, these would be connectors to
actual databases, but for simplicity, we just use dictionaries:

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}
```

The next step is to define the tools and mark them with `@mcp.tool()` so that they are discoverable by
MCP clients as MCP tools:

```python
@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights
```

The last step is spinning up the server:

```python
if __name__ == "__main__":
    mcp.run()
```

Now we have finished writing the server! Let's launch it:

```shell
python path_to_your_working_directory/mcp_server.py
```

## Write a DSPy Program That Utilizes Tools in MCP Server

Now that the server is running, let's build the actual airline service agent which
utilizes the MCP tools in our server to assist users. In your working directory,
create a file named `dspy_mcp_agent.py`, and follow the guide to add code to it.

### Gather Tools from MCP Servers

We first need to gather all available tools from the MCP server and make them
usable by DSPy. DSPy provides an API [`dspy.Tool`](https://dspy.ai/api/primitives/Tool/)
as the standard tool interface. Let's convert all the MCP tools to `dspy.Tool`.

We need to create an MCP client instance to communicate with the MCP server, fetch all available
tools, and convert them to `dspy.Tool` using the static method `from_mcp_tool`:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["path_to_your_working_directory/mcp_server.py"],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

With the code above, we have successfully collected all available MCP tools and converted
them to DSPy tools.


### Build a DSPy Agent to Handle Customer Requests

Now we will use `dspy.ReAct` to build the agent for handling customer requests. `ReAct` stands
for "reasoning and acting," which asks the LLM to decide whether to call a tool or wrap up the process.
If a tool is required, the LLM takes responsibility for deciding which tool to call and providing
the appropriate arguments.

As usual, we need to create a `dspy.Signature` to define the input and output of our agent:

```python
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests. You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )
```

And choose an LM for our agent:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

Then we create the ReAct agent by passing the tools and signature into the `dspy.ReAct` API. We can now
put together the complete code script:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["script_tmp/mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # Create the agent
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))
```

Note that we must call `react.acall` because MCP tools are async by default. Let's execute the script:

```shell
python path_to_your_working_directory/dspy_mcp_agent.py
```

You should see output similar to this:

```
Prediction(
    trajectory={'thought_0': 'I need to fetch flight information for Adam from SFO to JFK on 09/01/2025 to find available flights for booking.', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': ['{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}', '{"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}'], ..., 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},
    reasoning="I successfully booked a flight for Adam from SFO to JFK on 09/01/2025. I found two available flights, selected the more economical option (flight DA123 at 1 AM for $200), retrieved Adam's user profile, and completed the booking process. The confirmation number for the flight is 8h7clk3q.",
    process_result='Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 8h7clk3q.'
)
```

The `trajectory` field contains the entire thinking and acting process. If you're curious about what's happening
under the hood, check out the [Observability Guide](https://dspy.ai/tutorials/observability/) to set up MLflow,
which visualizes every step happening inside `dspy.ReAct`!


## Conclusion

In this guide, we built an airline service agent that utilizes a custom MCP server and the `dspy.ReAct` module. In the context
of MCP support, DSPy provides a simple interface for interacting with MCP tools, giving you the flexibility to implement
any functionality you need.
