"""
Shared Memory System Implementation for Chatterer-based Agent System

This module implements the shared memory system for agents, allowing them to:
1. Share information between agents
2. Control read/write access to shared memory
3. Efficiently exchange data without duplicating context
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum, auto

from ..language_model import Chatterer
from .delegation import DelegatingAgent, TaskContext

class MemoryAccessLevel(Enum):
    """Enum for memory access levels"""
    NO_ACCESS = auto()
    READ_ONLY = auto()
    WRITE_ONLY = auto()
    READ_WRITE = auto()

class MemoryEntry(BaseModel):
    """Model for a memory entry in shared memory"""
    key: str
    value: Any
    created_by: str
    created_at: float
    last_modified_by: Optional[str] = None
    last_modified_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SharedMemory:
    """Shared memory system for agents"""
    
    def __init__(self):
        self.memory: Dict[str, MemoryEntry] = {}
        self.access_control: Dict[str, Dict[str, MemoryAccessLevel]] = {}
        
    def set_agent_access(self, agent_id: str, access_level: MemoryAccessLevel) -> None:
        """Set default access level for an agent"""
        if agent_id not in self.access_control:
            self.access_control[agent_id] = {}
            
        # Set default access for all existing keys
        for key in self.memory.keys():
            self.access_control[agent_id][key] = access_level
    
    def set_key_access(self, agent_id: str, key: str, access_level: MemoryAccessLevel) -> None:
        """Set access level for a specific key for an agent"""
        if agent_id not in self.access_control:
            self.access_control[agent_id] = {}
            
        self.access_control[agent_id][key] = access_level
    
    def check_read_access(self, agent_id: str, key: str) -> bool:
        """Check if an agent has read access to a key"""
        if agent_id not in self.access_control:
            return False
            
        if key not in self.access_control[agent_id]:
            return False
            
        access = self.access_control[agent_id][key]
        return access in [MemoryAccessLevel.READ_ONLY, MemoryAccessLevel.READ_WRITE]
    
    def check_write_access(self, agent_id: str, key: str) -> bool:
        """Check if an agent has write access to a key"""
        if agent_id not in self.access_control:
            return False
            
        if key not in self.access_control[agent_id]:
            return False
            
        access = self.access_control[agent_id][key]
        return access in [MemoryAccessLevel.WRITE_ONLY, MemoryAccessLevel.READ_WRITE]
    
    def write(self, agent_id: str, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write a value to shared memory"""
        
        # Check write access
        if not self.check_write_access(agent_id, key):
            return False
            
        current_time = time.time()
        
        if key in self.memory:
            # Update existing entry
            entry = self.memory[key]
            entry.value = value
            entry.last_modified_by = agent_id
            entry.last_modified_at = current_time
            if metadata:
                entry.metadata.update(metadata)
        else:
            # Create new entry
            entry = MemoryEntry(
                key=key,
                value=value,
                created_by=agent_id,
                created_at=current_time,
                last_modified_by=agent_id,
                last_modified_at=current_time,
                metadata=metadata or {}
            )
            self.memory[key] = entry
            
            # Set access for all agents for this new key
            for agent_id, access_dict in self.access_control.items():
                # If agent doesn't have specific access for this key,
                # use their default access level if they have one
                if key not in access_dict and "*" in access_dict:
                    access_dict[key] = access_dict["*"]
        
        return True
    
    def read(self, agent_id: str, key: str) -> Optional[Any]:
        """Read a value from shared memory"""
        # Check read access
        if not self.check_read_access(agent_id, key):
            return None
            
        if key not in self.memory:
            return None
            
        return self.memory[key].value
    
    def get_all_readable(self, agent_id: str) -> Dict[str, Any]:
        """Get all readable memory entries for an agent"""
        result: Dict[str, Any] = {}
        
        for key in self.memory.keys():
            if self.check_read_access(agent_id, key):
                result[key] = self.memory[key].value
                
        return result
    
    def delete(self, agent_id: str, key: str) -> bool:
        """Delete a memory entry"""
        # Check write access (required for deletion)
        if not self.check_write_access(agent_id, key):
            return False
            
        if key in self.memory:
            del self.memory[key]
            
            # Remove access control entries for this key
            for agent_access in self.access_control.values():
                if key in agent_access:
                    del agent_access[key]
                    
            return True
            
        return False
    
    def get_memory_summary(self) -> str:
        """Get a summary of all memory entries"""
        if not self.memory:
            return "Shared memory is empty."
            
        summary = "Shared Memory Contents:\n"
        
        for key, entry in self.memory.items():
            summary += f"- {key}: {type(entry.value).__name__} (Created by: {entry.created_by})\n"
            
        return summary
    
    def get_access_summary(self) -> str:
        """Get a summary of access control"""
        if not self.access_control:
            return "No access control defined."
            
        summary = "Access Control:\n"
        
        for agent_id, access_dict in self.access_control.items():
            summary += f"Agent {agent_id}:\n"
            for key, access in access_dict.items():
                summary += f"  - {key}: {access.name}\n"
                
        return summary


class SharedMemoryAgent(DelegatingAgent):
    """Agent with shared memory capabilities"""
    
    def __init__(self, model: Chatterer, agent_id: Optional[str] = None, tools: Optional[List[Any]] = None,
                 delegation_enabled: bool = True, parent_agent: Optional['DelegatingAgent'] = None,
                 shared_memory: Optional[SharedMemory] = None, memory_access: MemoryAccessLevel = MemoryAccessLevel.NO_ACCESS):
        # Initialize with default values for None parameters
        actual_agent_id = agent_id if agent_id is not None else ""
        actual_tools = tools if tools is not None else []
        
        super().__init__(model, actual_agent_id, actual_tools, delegation_enabled, parent_agent)
        
        # Initialize shared memory
        self.shared_memory = shared_memory
        
        # Set up memory access if shared memory is provided
        if shared_memory:
            shared_memory.set_agent_access(self.agent_id, memory_access)
            
            # Register memory tools based on access level
            if memory_access in [MemoryAccessLevel.READ_ONLY, MemoryAccessLevel.READ_WRITE]:
                self.register_function_as_tool(
                    self.read_from_memory,
                    name="read_from_memory",
                    description="Read a value from shared memory"
                )
                self.register_function_as_tool(
                    self.list_memory_keys,
                    name="list_memory_keys",
                    description="List all readable memory keys"
                )
                
            if memory_access in [MemoryAccessLevel.WRITE_ONLY, MemoryAccessLevel.READ_WRITE]:
                self.register_function_as_tool(
                    self.write_to_memory,
                    name="write_to_memory",
                    description="Write a value to shared memory"
                )
                self.register_function_as_tool(
                    self.delete_from_memory,
                    name="delete_from_memory",
                    description="Delete a value from shared memory"
                )
    
    async def read_from_memory(self, key: str) -> Any:
        """Read a value from shared memory"""
        if not self.shared_memory:
            raise ValueError("Shared memory is not available")
            
        value = self.shared_memory.read(self.agent_id, key)
        if value is None:
            raise ValueError(f"Cannot read key '{key}' (either it doesn't exist or you don't have read access)")
            
        return value
    
    async def write_to_memory(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write a value to shared memory"""
        if not self.shared_memory:
            raise ValueError("Shared memory is not available")
            
        success = self.shared_memory.write(self.agent_id, key, value, metadata)
        if not success:
            raise ValueError(f"Cannot write to key '{key}' (you don't have write access)")
            
        return True
    
    async def delete_from_memory(self, key: str) -> bool:
        """Delete a value from shared memory"""
        if not self.shared_memory:
            raise ValueError("Shared memory is not available")
            
        success = self.shared_memory.delete(self.agent_id, key)
        if not success:
            raise ValueError(f"Cannot delete key '{key}' (either it doesn't exist or you don't have write access)")
            
        return True
    
    async def list_memory_keys(self) -> List[str]:
        """List all readable memory keys"""
        if not self.shared_memory:
            raise ValueError("Shared memory is not available")
            
        readable = self.shared_memory.get_all_readable(self.agent_id)
        return list(readable.keys())
    
    def create_sub_agent(self, sub_agent_id: Optional[str] = None, memory_access: Optional[MemoryAccessLevel] = None) -> 'SharedMemoryAgent':
        """Create a sub-agent with shared memory access"""
        # Generate sub-agent ID if not provided
        actual_sub_agent_id = sub_agent_id if sub_agent_id is not None else f"{self.agent_id}_sub_{len(self.sub_agents)}"
            
        # Use parent's memory access level if not specified
        actual_memory_access = memory_access
        if actual_memory_access is None and self.shared_memory:
            # Find this agent's default access level
            agent_access = self.shared_memory.access_control.get(self.agent_id, {})
            default_access = agent_access.get("*", MemoryAccessLevel.NO_ACCESS)
            actual_memory_access = default_access
        
        # Default to NO_ACCESS if still None
        if actual_memory_access is None:
            actual_memory_access = MemoryAccessLevel.NO_ACCESS
            
        # Create sub-agent
        sub_agent = SharedMemoryAgent(
            model=self.model,
            agent_id=actual_sub_agent_id,
            delegation_enabled=self.delegation_enabled,
            parent_agent=self,
            shared_memory=self.shared_memory,
            memory_access=actual_memory_access
        )
        
        # Register available tools for sub-agent
        for tool_name, tool in self.tools.items():
            # Skip memory tools as they'll be added based on access level
            if tool_name not in ["read_from_memory", "write_to_memory", "delete_from_memory", "list_memory_keys"]:
                sub_agent.register_tool(tool)
        
        # Store sub-agent
        self.sub_agents[actual_sub_agent_id] = sub_agent
        
        return sub_agent
    
    async def delegate_task(self, 
                           task_description: str, 
                           completion_criteria: str,
                           relevant_information: Optional[Dict[str, Any]] = None,
                           tool_names: Optional[List[str]] = None,
                           memory_access: Optional[str] = None) -> str:
        """
        Delegate a task to a new sub-agent with shared memory access
        
        Args:
            task_description: Detailed description of the task to be performed
            completion_criteria: Clear criteria for when the task is considered complete
            relevant_information: Dictionary of information relevant to the task
            tool_names: List of tool names the sub-agent should have access to
            memory_access: Memory access level for the sub-agent (NO_ACCESS, READ_ONLY, WRITE_ONLY, READ_WRITE)
        
        Returns:
            ID of the created sub-agent
        """
        if not self.delegation_enabled:
            raise ValueError("Delegation is not enabled for this agent")
        
        # Parse memory access level
        memory_access_level = None
        if memory_access:
            try:
                memory_access_level = MemoryAccessLevel[memory_access]
            except KeyError:
                raise ValueError(f"Invalid memory access level: {memory_access}")
        
        # Create task context
        context = TaskContext(
            task_description=task_description,
            completion_criteria=completion_criteria,
            relevant_information=relevant_information or {},
            available_tools=tool_names or [name for name in self.tools.keys() 
                                          if name not in ["read_from_memory", "write_to_memory", 
                                                         "delete_from_memory", "list_memory_keys"]],
            parent_agent_id=self.agent_id
        )
        
        # Create sub-agent
        sub_agent_id = f"{self.agent_id}_sub_{len(self.sub_agents)}"
        sub_agent = self.create_sub_agent(sub_agent_id, memory_access_level)
        
        # Create task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_delegated_task(sub_agent, task_id, context))
        
        return sub_agent_id
    
    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt, allowing it to use tools, delegate tasks, and access shared memory"""
        # Create system message with tool descriptions, delegation capabilities, and shared memory info
        memory_info = ""
        if self.shared_memory:
            readable_keys = await self.list_memory_keys()
            if readable_keys:
                memory_info = "\nYou have access to shared memory with the following keys:\n"
                memory_info += ", ".join(readable_keys)
                memory_info += "\nUse the memory tools to read from or write to shared memory."
        
        system_message = f"""You are an AI assistant that can use tools, delegate tasks to other agents, and access shared memory.
Available tools:

{self.get_tool_descriptions()}

To use a tool, respond with:
```tool
{{"tool_name": "<tool_name>", "parameters": {{"param1": "value1", "param2": "value2"}}}}
```

You can delegate complex tasks to other agents using the delegate_task tool.
When delegating, provide clear task description, completion criteria, and relevant information.
{memory_info}

After using a tool, you'll receive the result and can use it to continue your task.
You can use multiple tools by responding with multiple tool blocks.
When you're done, respond with your final answer without any tool blocks.
"""
        
        # Initialize conversation
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Maximum number of tool use iterations to prevent infinite loops
        max_iterations = 15
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get response from model
            response = await self.model.agenerate(conversation)
            
            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": response})
            
            # Check if response contains tool use
            if "```tool" in response:
                # Extract tool use blocks
                tool_blocks = response.split("```tool")
                
                # Process each tool block
                for block in tool_blocks[1:]:
                    # Extract JSON part
                    json_str = block.split("```")[0].strip()
                    
                    try:
                        tool_call = json.loads(json_str)
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})
                        
                        # Execute tool
                        tool_result = await self.execute_tool(tool_name, **parameters)
                        
                        # Add tool result to conversation
                        result_str = f"Tool result: {json.dumps(tool_result, indent=2)}"
                        conversation.append({"role": "system", "content": result_str})
                        
                    except Exception as e:
                        # Add error to conversation
                        error_str = f"Error executing tool: {str(e)}"
                        conversation.append({"role": "system", "content": error_str})
            else:
                # No tool use, return final response
                return response
        
        # If maximum iterations reached, return a message
        return "Maximum number of tool use iterations reached. Please try a simpler request or break it down into smaller steps."
