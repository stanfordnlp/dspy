# Building a Creative Text-Based AI Game with DSPy

This tutorial demonstrates how to create an interactive text-based adventure game using DSPy's modular programming approach. You'll build a dynamic game where AI handles narrative generation, character interactions, and adaptive gameplay.

## What You'll Build

An intelligent text-based adventure game featuring:

- Dynamic story generation and branching narratives
- AI-powered character interactions and dialogue
- Adaptive gameplay that responds to player choices
- Inventory and character progression systems
- Save/load game state functionality

## Setup

```bash
pip install dspy rich typer
```

## Step 1: Core Game Framework

```python
import dspy
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import typer

# Configure DSPy
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

console = Console()

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    INVENTORY = "inventory"
    CHARACTER = "character"
    GAME_OVER = "game_over"

@dataclass
class Player:
    name: str
    health: int = 100
    level: int = 1
    experience: int = 0
    inventory: list[str] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=lambda: {
        "strength": 10,
        "intelligence": 10,
        "charisma": 10,
        "stealth": 10
    })
    
    def add_item(self, item: str):
        self.inventory.append(item)
        console.print(f"[green]Added {item} to inventory![/green]")
    
    def remove_item(self, item: str) -> bool:
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False
    
    def gain_experience(self, amount: int):
        self.experience += amount
        old_level = self.level
        self.level = 1 + (self.experience // 100)
        if self.level > old_level:
            console.print(f"[bold yellow]Level up! You are now level {self.level}![/bold yellow]")

@dataclass
class GameContext:
    current_location: str = "Village Square"
    story_progress: int = 0
    visited_locations: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    game_flags: dict[str, bool] = field(default_factory=dict)
    
    def add_flag(self, flag: str, value: bool = True):
        self.game_flags[flag] = value
    
    def has_flag(self, flag: str) -> bool:
        return self.game_flags.get(flag, False)

class GameEngine:
    def __init__(self):
        self.player = None
        self.context = GameContext()
        self.state = GameState.MENU
        self.running = True
        
    def save_game(self, filename: str = "savegame.json"):
        """Save current game state."""
        save_data = {
            "player": {
                "name": self.player.name,
                "health": self.player.health,
                "level": self.player.level,
                "experience": self.player.experience,
                "inventory": self.player.inventory,
                "skills": self.player.skills
            },
            "context": {
                "current_location": self.context.current_location,
                "story_progress": self.context.story_progress,
                "visited_locations": self.context.visited_locations,
                "npcs_met": self.context.npcs_met,
                "completed_quests": self.context.completed_quests,
                "game_flags": self.context.game_flags
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        console.print(f"[green]Game saved to {filename}![/green]")
    
    def load_game(self, filename: str = "savegame.json") -> bool:
        """Load game state from file."""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            # Reconstruct player
            player_data = save_data["player"]
            self.player = Player(
                name=player_data["name"],
                health=player_data["health"],
                level=player_data["level"],
                experience=player_data["experience"],
                inventory=player_data["inventory"],
                skills=player_data["skills"]
            )
            
            # Reconstruct context
            context_data = save_data["context"]
            self.context = GameContext(
                current_location=context_data["current_location"],
                story_progress=context_data["story_progress"],
                visited_locations=context_data["visited_locations"],
                npcs_met=context_data["npcs_met"],
                completed_quests=context_data["completed_quests"],
                game_flags=context_data["game_flags"]
            )
            
            console.print(f"[green]Game loaded from {filename}![/green]")
            return True
            
        except FileNotFoundError:
            console.print(f"[red]Save file {filename} not found![/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error loading game: {e}![/red]")
            return False

# Initialize game engine
game = GameEngine()
```

## Step 2: AI-Powered Story Generation

```python
class StoryGenerator(dspy.Signature):
    """Generate dynamic story content based on current game state."""
    location: str = dspy.InputField(desc="Current location")
    player_info: str = dspy.InputField(desc="Player information and stats")
    story_progress: int = dspy.InputField(desc="Current story progress level")
    recent_actions: str = dspy.InputField(desc="Player's recent actions")
    
    scene_description: str = dspy.OutputField(desc="Vivid description of current scene")
    available_actions: list[str] = dspy.OutputField(desc="List of possible player actions")
    npcs_present: list[str] = dspy.OutputField(desc="NPCs present in this location")
    items_available: list[str] = dspy.OutputField(desc="Items that can be found or interacted with")

class DialogueGenerator(dspy.Signature):
    """Generate NPC dialogue and responses."""
    npc_name: str = dspy.InputField(desc="Name and type of NPC")
    npc_personality: str = dspy.InputField(desc="NPC personality and background")
    player_input: str = dspy.InputField(desc="What the player said or did")
    context: str = dspy.InputField(desc="Current game context and history")
    
    npc_response: str = dspy.OutputField(desc="NPC's dialogue response")
    mood_change: str = dspy.OutputField(desc="How NPC's mood changed (positive/negative/neutral)")
    quest_offered: bool = dspy.OutputField(desc="Whether NPC offers a quest")
    information_revealed: str = dspy.OutputField(desc="Any important information shared")

class ActionResolver(dspy.Signature):
    """Resolve player actions and determine outcomes."""
    action: str = dspy.InputField(desc="Player's chosen action")
    player_stats: str = dspy.InputField(desc="Player's current stats and skills")
    context: str = dspy.InputField(desc="Current game context")
    difficulty: str = dspy.InputField(desc="Difficulty level of the action")
    
    success: bool = dspy.OutputField(desc="Whether the action succeeded")
    outcome_description: str = dspy.OutputField(desc="Description of what happened")
    stat_changes: dict[str, int] = dspy.OutputField(desc="Changes to player stats")
    items_gained: list[str] = dspy.OutputField(desc="Items gained from this action")
    experience_gained: int = dspy.OutputField(desc="Experience points gained")

class GameAI(dspy.Module):
    """Main AI module for game logic and narrative."""
    
    def __init__(self):
        super().__init__()
        self.story_gen = dspy.ChainOfThought(StoryGenerator)
        self.dialogue_gen = dspy.ChainOfThought(DialogueGenerator)
        self.action_resolver = dspy.ChainOfThought(ActionResolver)
    
    def generate_scene(self, player: Player, context: GameContext, recent_actions: str = "") -> Dict:
        """Generate current scene description and options."""
        
        player_info = f"Level {player.level} {player.name}, Health: {player.health}, Skills: {player.skills}"
        
        scene = self.story_gen(
            location=context.current_location,
            player_info=player_info,
            story_progress=context.story_progress,
            recent_actions=recent_actions
        )
        
        return {
            "description": scene.scene_description,
            "actions": scene.available_actions,
            "npcs": scene.npcs_present,
            "items": scene.items_available
        }
    
    def handle_dialogue(self, npc_name: str, player_input: str, context: GameContext) -> Dict:
        """Handle conversation with NPCs."""
        
        # Create NPC personality based on name and context
        personality_map = {
            "Village Elder": "Wise, knowledgeable, speaks in riddles, has ancient knowledge",
            "Merchant": "Greedy but fair, loves to bargain, knows about valuable items",
            "Guard": "Dutiful, suspicious of strangers, follows rules strictly",
            "Thief": "Sneaky, untrustworthy, has information about hidden things",
            "Wizard": "Mysterious, powerful, speaks about magic and ancient forces"
        }
        
        personality = personality_map.get(npc_name, "Friendly villager with local knowledge")
        game_context = f"Location: {context.current_location}, Story progress: {context.story_progress}"
        
        response = self.dialogue_gen(
            npc_name=npc_name,
            npc_personality=personality,
            player_input=player_input,
            context=game_context
        )
        
        return {
            "response": response.npc_response,
            "mood": response.mood_change,
            "quest": response.quest_offered,
            "info": response.information_revealed
        }
    
    def resolve_action(self, action: str, player: Player, context: GameContext) -> Dict:
        """Resolve player actions and determine outcomes."""
        
        player_stats = f"Level {player.level}, Health {player.health}, Skills: {player.skills}"
        game_context = f"Location: {context.current_location}, Progress: {context.story_progress}"
        
        # Determine difficulty based on action type
        difficulty = "medium"
        if any(word in action.lower() for word in ["fight", "battle", "attack"]):
            difficulty = "hard"
        elif any(word in action.lower() for word in ["look", "examine", "talk"]):
            difficulty = "easy"
        
        result = self.action_resolver(
            action=action,
            player_stats=player_stats,
            context=game_context,
            difficulty=difficulty
        )
        
        return {
            "success": result.success,
            "description": result.outcome_description,
            "stat_changes": result.stat_changes,
            "items": result.items_gained,
            "experience": result.experience_gained
        }

# Initialize AI
ai = GameAI()
```

## Step 3: Game Interface and Interaction

```python
def display_game_header():
    """Display the game header."""
    header = Text("🏰 MYSTIC REALM ADVENTURE 🏰", style="bold magenta")
    console.print(Panel(header, style="bright_blue"))

def display_player_status(player: Player):
    """Display player status panel."""
    status = f"""
[bold]Name:[/bold] {player.name}
[bold]Level:[/bold] {player.level} (XP: {player.experience})
[bold]Health:[/bold] {player.health}/100
[bold]Skills:[/bold]
  • Strength: {player.skills['strength']}
  • Intelligence: {player.skills['intelligence']}
  • Charisma: {player.skills['charisma']}
  • Stealth: {player.skills['stealth']}
[bold]Inventory:[/bold] {len(player.inventory)} items
    """
    console.print(Panel(status.strip(), title="Player Status", style="green"))

def display_location(context: GameContext, scene: Dict):
    """Display current location and scene."""
    location_panel = f"""
[bold yellow]{context.current_location}[/bold yellow]

{scene['description']}
    """
    
    if scene['npcs']:
        location_panel += f"\n\n[bold]NPCs present:[/bold] {', '.join(scene['npcs'])}"
    
    if scene['items']:
        location_panel += f"\n[bold]Items visible:[/bold] {', '.join(scene['items'])}"
    
    console.print(Panel(location_panel.strip(), title="Current Location", style="cyan"))

def display_actions(actions: list[str]):
    """Display available actions."""
    action_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
    console.print(Panel(action_text, title="Available Actions", style="yellow"))

def get_player_choice(max_choices: int) -> int:
    """Get player's choice with input validation."""
    while True:
        try:
            choice = typer.prompt("Choose an action (number)")
            choice_num = int(choice)
            if 1 <= choice_num <= max_choices:
                return choice_num - 1
            else:
                console.print(f"[red]Please enter a number between 1 and {max_choices}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

def show_inventory(player: Player):
    """Display player inventory."""
    if not player.inventory:
        console.print(Panel("Your inventory is empty.", title="Inventory", style="red"))
    else:
        items = "\n".join([f"• {item}" for item in player.inventory])
        console.print(Panel(items, title="Inventory", style="green"))

def main_menu():
    """Display main menu and handle selection."""
    console.clear()
    display_game_header()
    
    menu_options = [
        "1. New Game",
        "2. Load Game", 
        "3. How to Play",
        "4. Exit"
    ]
    
    menu_text = "\n".join(menu_options)
    console.print(Panel(menu_text, title="Main Menu", style="bright_blue"))
    
    choice = typer.prompt("Select an option")
    return choice

def show_help():
    """Display help information."""
    help_text = """
[bold]How to Play:[/bold]

• This is a text-based adventure game powered by AI
• Make choices by selecting numbered options
• Talk to NPCs to learn about the world and get quests
• Explore different locations to find items and adventures
• Your choices affect the story and character development
• Use 'inventory' to check your items
• Use 'status' to see your character info
• Type 'save' to save your progress
• Type 'quit' to return to main menu

[bold]Tips:[/bold]
• Different skills affect your success in various actions
• NPCs remember your previous interactions
• Explore thoroughly - there are hidden secrets!
• Your reputation affects how NPCs treat you
    """
    console.print(Panel(help_text.strip(), title="Game Help", style="blue"))
    typer.prompt("Press Enter to continue")
```

## Step 4: Main Game Loop

```python
def create_new_character():
    """Create a new player character."""
    console.clear()
    display_game_header()
    
    name = typer.prompt("Enter your character's name")
    
    # Character creation with skill point allocation
    console.print("\n[bold]Character Creation[/bold]")
    console.print("You have 10 extra skill points to distribute among your skills.")
    console.print("Base skills start at 10 each.\n")
    
    skills = {"strength": 10, "intelligence": 10, "charisma": 10, "stealth": 10}
    points_remaining = 10
    
    for skill in skills.keys():
        if points_remaining > 0:
            console.print(f"Points remaining: {points_remaining}")
            while True:
                try:
                    points = int(typer.prompt(f"Points to add to {skill} (0-{points_remaining})"))
                    if 0 <= points <= points_remaining:
                        skills[skill] += points
                        points_remaining -= points
                        break
                    else:
                        console.print(f"[red]Enter a number between 0 and {points_remaining}[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number[/red]")
    
    player = Player(name=name, skills=skills)
    console.print(f"\n[green]Welcome to Mystic Realm, {name}![/green]")
    return player

def game_loop():
    """Main game loop."""
    recent_actions = ""
    
    while game.running and game.state == GameState.PLAYING:
        console.clear()
        display_game_header()
        
        # Generate current scene
        scene = ai.generate_scene(game.player, game.context, recent_actions)
        
        # Display game state
        display_player_status(game.player)
        display_location(game.context, scene)
        
        # Add standard actions
        all_actions = scene['actions'] + ["Check inventory", "Character status", "Save game", "Quit to menu"]
        display_actions(all_actions)
        
        # Get player choice
        choice_idx = get_player_choice(len(all_actions))
        chosen_action = all_actions[choice_idx]
        
        # Handle special commands
        if chosen_action == "Check inventory":
            show_inventory(game.player)
            typer.prompt("Press Enter to continue")
            continue
        elif chosen_action == "Character status":
            display_player_status(game.player)
            typer.prompt("Press Enter to continue")
            continue
        elif chosen_action == "Save game":
            game.save_game()
            typer.prompt("Press Enter to continue")
            continue
        elif chosen_action == "Quit to menu":
            game.state = GameState.MENU
            break
        
        # Handle game actions
        if chosen_action in scene['actions']:
            # Check if it's dialogue with an NPC
            npc_target = None
            for npc in scene['npcs']:
                if npc.lower() in chosen_action.lower():
                    npc_target = npc
                    break
            
            if npc_target:
                # Handle NPC interaction
                console.print(f"\n[bold]Talking to {npc_target}...[/bold]")
                dialogue = ai.handle_dialogue(npc_target, chosen_action, game.context)
                
                console.print(f"\n[italic]{npc_target}:[/italic] \"{dialogue['response']}\"")
                
                if dialogue['quest']:
                    console.print(f"[yellow]💼 Quest opportunity detected![/yellow]")
                
                if dialogue['info']:
                    console.print(f"[blue]ℹ️  {dialogue['info']}[/blue]")
                    
                # Add NPC to met list
                if npc_target not in game.context.npcs_met:
                    game.context.npcs_met.append(npc_target)
                
                recent_actions = f"Talked to {npc_target}: {chosen_action}"
            else:
                # Handle general action
                result = ai.resolve_action(chosen_action, game.player, game.context)
                
                console.print(f"\n{result['description']}")
                
                # Apply results
                if result['success']:
                    console.print("[green]✅ Success![/green]")
                    
                    # Apply stat changes
                    for stat, change in result['stat_changes'].items():
                        if stat in game.player.skills:
                            game.player.skills[stat] += change
                            if change > 0:
                                console.print(f"[green]{stat.title()} increased by {change}![/green]")
                        elif stat == "health":
                            game.player.health = max(0, min(100, game.player.health + change))
                            if change > 0:
                                console.print(f"[green]Health restored by {change}![/green]")
                            elif change < 0:
                                console.print(f"[red]Health decreased by {abs(change)}![/red]")
                    
                    # Add items
                    for item in result['items']:
                        game.player.add_item(item)
                    
                    # Give experience
                    if result['experience'] > 0:
                        game.player.gain_experience(result['experience'])
                    
                    # Update story progress
                    game.context.story_progress += 1
                else:
                    console.print("[red]❌ The action didn't go as planned...[/red]")
                
                recent_actions = f"Attempted: {chosen_action}"
            
            # Check for game over conditions
            if game.player.health <= 0:
                console.print("\n[bold red]💀 You have died! Game Over![/bold red]")
                game.state = GameState.GAME_OVER
                break
            
            typer.prompt("\nPress Enter to continue")

def main():
    """Main game function."""
    while game.running:
        if game.state == GameState.MENU:
            choice = main_menu()
            
            if choice == "1":
                game.player = create_new_character()
                game.context = GameContext()
                game.state = GameState.PLAYING
                console.print("\n[italic]Your adventure begins...[/italic]")
                typer.prompt("Press Enter to start")
                
            elif choice == "2":
                if game.load_game():
                    game.state = GameState.PLAYING
                typer.prompt("Press Enter to continue")
                
            elif choice == "3":
                show_help()
                
            elif choice == "4":
                game.running = False
                console.print("[bold]Thanks for playing! Goodbye![/bold]")
            
        elif game.state == GameState.PLAYING:
            game_loop()
            
        elif game.state == GameState.GAME_OVER:
            console.print("\n[bold]Game Over[/bold]")
            restart = typer.confirm("Would you like to return to the main menu?")
            if restart:
                game.state = GameState.MENU
            else:
                game.running = False

if __name__ == "__main__":
    main()
```

## Example Gameplay

When you run the game, you'll experience:

**Character Creation:**
```
🏰 MYSTIC REALM ADVENTURE 🏰

Enter your character's name: Aria

Character Creation
You have 10 extra skill points to distribute among your skills.
Base skills start at 10 each.

Points remaining: 10
Points to add to strength (0-10): 2
Points to add to intelligence (0-8): 4
Points to add to charisma (0-4): 3
Points to add to stealth (0-1): 1

Welcome to Mystic Realm, Aria!
```

**Dynamic Scene Generation:**
```
┌──────────── Current Location ────────────┐
│ Village Square                           │
│                                          │
│ You stand in the bustling heart of       │
│ Willowbrook Village. The ancient stone   │
│ fountain bubbles cheerfully as merchants │
│ hawk their wares and children play. A    │
│ mysterious hooded figure lurks near the  │
│ shadows of the old oak tree.             │
│                                          │
│ NPCs present: Village Elder, Merchant    │
│ Items visible: Strange Medallion, Herbs  │
└──────────────────────────────────────────┘

┌────────── Available Actions ─────────────┐
│ 1. Approach the hooded figure            │
│ 2. Talk to the Village Elder             │
│ 3. Browse the merchant's wares           │
│ 4. Examine the strange medallion         │
│ 5. Gather herbs near the fountain        │
│ 6. Head to the forest path               │
└───────────────────────────────────────────┘
```

**AI-Generated Dialogue:**
```
Talking to Village Elder...

Village Elder: "Ah, young traveler, I sense a great destiny 
surrounds you like morning mist. The ancient prophecy speaks 
of one who would come bearing the mark of courage. Tell me, 
have you noticed anything... unusual in your travels?"

💼 Quest opportunity detected!
ℹ️ The Village Elder knows about an ancient prophecy that might involve you
```

## Next Steps

- **Combat System**: Add turn-based battles with strategy
- **Magic System**: Spellcasting with resource management
- **Multiplayer**: Network support for cooperative adventures
- **Quest System**: Complex multi-step missions with branching outcomes
- **World Building**: Procedurally generated locations and characters
- **Audio**: Add sound effects and background music

This tutorial demonstrates how DSPy's modular approach enables complex, interactive systems where AI handles creative content generation while maintaining consistent game logic and player agency.
