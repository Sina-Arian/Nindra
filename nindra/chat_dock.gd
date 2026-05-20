@tool
extends MarginContainer

# ----------------- UI nodes ----------------------
@onready var http_request = $HTTPRequest
@onready var message_display = $VBoxContainer/MessageDisplay
@onready var prompt_input = $VBoxContainer/HBoxContainer/PromptInput
@onready var send_button = $VBoxContainer/HBoxContainer/SendButton
@onready var clear_button = $VBoxContainer/HBoxContainer/ClearButton
@onready var embed_button: Button = $VBoxContainer/HBoxContainer/EmbedButton
@onready var suggest_next_button = $VBoxContainer/HBoxContainer/SuggestNextButton
@onready var improve_button = $VBoxContainer/HBoxContainer/ImproveButton
@onready var avatar_texture = $VBoxContainer/TextureRect

# ----------------- Constants ---------------------
const OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
const EMBED_URL = "http://127.0.0.1:11434/api/embed"
const MODEL = "phi4-mini:3.8b-q8_0"
const EMBED_MODEL = "mxbai-embed-large:335m-v1-fp16"
const CONTEXT_FILE_PATH = "res://context_export.txt"
const MAX_HISTORY_PAIRS = 5            # keeps the last 5 user‑ai exchanges
const RETRIEVAL_TOP_K = 5              # how many chunks to inject as context
const CHUNK_MAX_CHARS = 1024            # ensure each chunk fits embedding model's token limit, text exceeding this limit is truncated before processing, which could lead to a loss of information
const MMR_LAMBDA: float = 0.7          # trade‑off: 1.0 = pure relevance(pure cosine similarity), 0.0 = pure diversity

# ----------------- Variables ---------------------
var conversation := []                 # visible chat history
var _conversation_display_text := ""   # Holds the complete BBCode conversation without streaming
var _last_warning := ""                # avoid spam warnings
var _last_streaming_text := ""

#------------------ Streaming state----------------
var _streaming_client: HTTPClient = null
var _streaming_response_text := ""
var _is_streaming := false
var _streaming_display_prefix := ""
var _stream_done := false
var _skip_history_next := false  # We'll add a temporary flag to avoid polluting conversation with code tasks
var _stream_buffer := ""         # Accumulates partial lines

#--------------------RAG state-------------------
var _chunks: PackedStringArray = []                    # text fragments of the project
var _chunk_embeddings: Array[PackedFloat32Array] = []  # one vector per chunk
var _indexed := false                                  # true after successful indexing
var _embedding_in_progress := false                    # avoid overlapping requests
var _pending_user_text := ""                           # holds user query while retrieval runs

# Separate HTTPRequest node for embedding calls so we don't mix with chat
var embed_request: HTTPRequest = null

func _ready():
	# ----------- Create a dedicated HTTPRequest for embeddings ----------------
	embed_request = HTTPRequest.new()
	add_child(embed_request)
	embed_request.request_completed.connect(_on_embed_request_completed)
	
	#----------------------- Wire UI signals------------------------------------
	if not embed_button.pressed.is_connected(_on_embed_pressed):
		embed_button.pressed.connect(_on_embed_pressed)
	if not send_button.pressed.is_connected(_on_send_pressed):
		send_button.pressed.connect(_on_send_pressed)
	if not clear_button.pressed.is_connected(_on_clear_pressed):
		clear_button.pressed.connect(_on_clear_pressed)
	if not suggest_next_button.pressed.is_connected(_on_suggest_next_pressed):
		suggest_next_button.pressed.connect(_on_suggest_next_pressed)
	if not improve_button.pressed.is_connected(_on_improve_pressed):
		improve_button.pressed.connect(_on_improve_pressed)
	if not http_request.request_completed.is_connected(_on_http_request_request_completed):
		http_request.request_completed.connect(_on_http_request_request_completed)
	avatar_texture.gui_input.connect(_on_avatar_gui_input)
	
	# Load your custom font (adjust the path to your own file)
	var custom_font = load("res://addons/nindra/fonts/CascadiaCodePL.ttf")
	if custom_font:
		message_display.add_theme_font_override("normal_font", custom_font)
	# Optional: also override bold, italics, etc.
	# message_display.add_theme_font_override("bold_font", custom_font)
	
	embed_button.text = "Index Project"
	set_process(true)

# ==============================================================================
#   STREAMING POLLING (unchanged from original)
# ==============================================================================
func _process(delta: float) -> void:
	if not _is_streaming:
		return
	_poll_streaming()


func _poll_streaming():
	if _streaming_client == null:
		_is_streaming = false
		send_button.disabled = false
		return
	
	_streaming_client.poll()
	var status = _streaming_client.get_status()
	#print("[DEBUG] Streaming status: ", status)
	
	match status:
		HTTPClient.STATUS_CONNECTING, HTTPClient.STATUS_RESOLVING:
			# Still working on the connection
			pass
		
		HTTPClient.STATUS_CONNECTED:
			if _streaming_state == 1:   # just connected, send the request
				var err = _streaming_client.request(
					HTTPClient.METHOD_POST,
					_streaming_request_path,
					_streaming_request_headers,
					_streaming_request_body
				)
				if err != OK:
					message_display.append_text("[color=red]Request failed: %s[/color]\n" % error_string(err))
					_finish_streaming()
				else:
					_streaming_state = 3   # now requesting
		
		HTTPClient.STATUS_REQUESTING:
			# Waiting for headers
			pass
		
		HTTPClient.STATUS_BODY:
			# Read every available chunk while there is data
			while true:
				var chunk = _streaming_client.read_response_body_chunk()
				if chunk.size() == 0:
					break          # no more data this frame
				_process_stream_chunk(chunk)
			# If the connection is still open, we'll be called again next frame.
			
			# Stop waiting for a disconnect – end the stream now
			if _stream_done:
				_finish_streaming()
		
		HTTPClient.STATUS_DISCONNECTED:
			# All handling is now done in _finish_streaming() – nothing to do here.
			pass
		_:
			#print("[DEBUG] Unhandled status: ", status)
			pass


func _process_stream_chunk(chunk: PackedByteArray):
	# Append new data to the buffer
	_stream_buffer += chunk.get_string_from_utf8()
	
	# Split buffer into lines (returns PackedStringArray)
	var packed_lines = _stream_buffer.split("\n")
	# Convert to regular Array so we can use pop_back()
	var lines = Array(packed_lines)
	
	# The last element may be incomplete (no trailing newline) – extract and remove it
	var last_partial = ""
	if lines.size() > 0:
		last_partial = lines[-1]      # get last element
		lines.remove_at(lines.size() - 1)  # remove it from the array
	
	# Process each complete line
	for line in lines:
		if line.strip_edges() == "":
			continue
		var json = JSON.new()
		var err = json.parse(line)
		if err != OK:
			print("JSON parse error: ", line)
			continue
		var data = json.get_data()
		if data.has("message") and data["message"].has("content"):
			var content = data["message"]["content"]
			_streaming_response_text += content
			_update_streaming_display()
		if data.has("done") and data["done"] == true:
			_stream_done = true
	
	# Keep the incomplete line for the next chunk
	_stream_buffer = last_partial


func _update_streaming_display():
	if _streaming_response_text == _last_streaming_text:
		return # nothing new, skip
	_last_streaming_text = _streaming_response_text
	var display = _conversation_display_text + _streaming_display_prefix + _streaming_response_text
	# Show a blinking cursor only if there is no text yet
	if _streaming_response_text == "":
		display += "[color=gray]▊[/color]"
	message_display.text = display
	message_display.scroll_to_line(message_display.get_line_count() - 1)


func _finish_streaming():
	# Finalise the assistant message (if any content was received)
	if _streaming_response_text.strip_edges() != "":
		var final_text = _streaming_response_text.strip_edges()
		
		# Only add to conversation history for normal chats (not code tasks)
		# (streaming is only used when _skip_history_next is false, but keep the check)
		if not _skip_history_next:
			conversation.append({"role": "assistant", "content": final_text})
			_trim_conversation()
		
		# Convert markdown to BBCode and permanently append to the display text
		var formatted := _markdown_to_bbcode(final_text)
		_conversation_display_text += "🤖 Nindra:\n" + formatted + "\n\n"
	
	# Show the complete conversation (no streaming suffix)
	message_display.text = _conversation_display_text
	message_display.scroll_to_line(message_display.get_line_count() - 1)
	
	# Clean up streaming state
	_is_streaming = false
	_streaming_state = 0
	_streaming_response_text = ""        
	_streaming_display_prefix = ""
	_stream_done = false
	_last_streaming_text = ""
	
	if _streaming_client:
		_streaming_client.close()
		_streaming_client = null
	
	send_button.disabled = false


# ==============================================================================
#   CHUNKING LOGIC
# ==============================================================================

# Main entry point: splits the entire export file content into chunks.
func _chunk_code(content: String) -> PackedStringArray:
	var chunks = PackedStringArray()
	
	# 1. Extract the three major sections
	var sections = _extract_sections(content, [
		"--- AUTOLOADS / GLOBALS ---",
		"--- SCRIPTS ---",
		"--- SCENES ---"
	])
	
	for section in sections:
		if section.begins_with("--- SCRIPTS ---"):
			chunks.append_array(_chunk_scripts_section(section))
		elif section.begins_with("--- SCENES ---"):
			chunks.append_array(_chunk_scenes_section(section))
		else:  # AUTOLOADS / GLOBALS
			chunks.append_array(_chunk_autoloads_section(section))
	
	return chunks


# Splits content by known section headers. Returns array of section strings (including the header line).
func _extract_sections(content: String, headers: PackedStringArray) -> PackedStringArray:
	var sections = PackedStringArray()
	var pos = 0
	for i in range(headers.size()):
		var start = content.find(headers[i], pos)
		if start == -1:
			continue
		var end = content.find(headers[i+1], start) if i+1 < headers.size() else content.length()
		sections.append(content.substr(start, end - start).strip_edges())
		pos = end
	
	return sections


# Handles "--- SCRIPTS ---" section: finds each script block and chunks its code.
func _chunk_scripts_section(section: String) -> PackedStringArray:
	var chunks = PackedStringArray()
	var script_regex = RegEx.new()
	# Matches: "--- SCRIPT: path ---" then newline, then ```gdscript ... ```
	script_regex.compile("(?s)--- SCRIPT: ([^\\n]+) ---\\s*```gdscript\\s*(.*?)\\s*```")
	for match in script_regex.search_all(section):
		var script_path = match.get_string(1)
		var code = match.get_string(2)
		var script_chunks = _chunk_gdscript(code, script_path)
		chunks.append_array(script_chunks)
	return chunks


# Handles "--- SCENES ---" section: splits by scene file headers.
func _chunk_scenes_section(section: String) -> PackedStringArray:
	var chunks = PackedStringArray()
	var scene_regex = RegEx.new()
	# Matches: "scene_name.tscn:" at start of line, then the indented tree until next scene or end
	scene_regex.compile("(?s)(?m)^([^\\n]+\\.tscn):\\n(.*?)(?=\\n[^\\s]+\\.tscn:|$)")
	for match in scene_regex.search_all(section):
		var scene_name = match.get_string(1)
		var tree = match.get_string(2)
		var full_scene = "%s:\n%s" % [scene_name, tree]
		if full_scene.length() > CHUNK_MAX_CHARS:
			chunks.append_array(_split_by_fixed_size(full_scene))
		else:
			chunks.append(full_scene)
	return chunks



# Handles "--- AUTOLOADS / GLOBALS ---" section.
# This section contains a GDScript (global.gd) and a scene tree (sound_manager.tscn).
func _chunk_autoloads_section(section: String) -> PackedStringArray:
	var chunks = PackedStringArray()
	
	# First, extract the GDScript block (between ```gdscript and ```)
	var script_regex = RegEx.new()
	script_regex.compile("(?s)```gdscript\\s*(.*?)\\s*```")
	var script_match = script_regex.search(section)
	if script_match:
		var code = script_match.get_string(1)
		chunks.append_array(_chunk_gdscript(code, "global.gd"))
	
	var text_regex = RegEx.new()
	text_regex.compile("(?s)```text\\s*(.*?)\\s*```")
	var text_match = text_regex.search(section)
	if text_match:
		var tree_text = text_match.get_string(1)
		# Now parse the tree_text for .tscn: lines (they appear as plain text inside)
		var scene_regex = RegEx.new()
		scene_regex.compile("(?s)(?m)^([^\\n]+\\.tscn):\\n(.*?)(?=\\n[^\\s]+\\.tscn:|$)")
		for scene_match in scene_regex.search_all(tree_text):
			var scene_name = scene_match.get_string(1)
			var tree = scene_match.get_string(2)
			var full_scene = "%s:\n%s" % [scene_name, tree]
			if full_scene.length() > CHUNK_MAX_CHARS:
				chunks.append_array(_split_by_fixed_size(full_scene))
			else:
				chunks.append(full_scene)
	
	return chunks


# Splits a GDScript code string into chunks by top‑level declarations.
# Preserves function signatures, class headers, etc.
func _chunk_gdscript(code: String, context: String = "") -> PackedStringArray:
	var chunks = PackedStringArray()
	var lines = code.split("\n")
	var current_chunk = ""
	var in_multiline = false
	var indent_stack = 0
	
	for line in lines:
		var stripped = line.strip_edges()
		var leading_spaces = line.length() - line.lstrip(" ").length()
		
		# Detect start of a new top‑level declaration (indent 0, not inside multiline)
		if not in_multiline and leading_spaces == 0 and not stripped.is_empty():
			# If current chunk would exceed limit, save it and start new
			if current_chunk.length() + line.length() + 1 > CHUNK_MAX_CHARS and not current_chunk.is_empty():
				chunks.append(_trim_chunk(current_chunk.strip_edges(), context))
				current_chunk = line + "\n"
			else:
				current_chunk += line + "\n"
		else:
			current_chunk += line + "\n"
		
		# Track multiline strings / parentheses (simplistic but works for most cases)
		in_multiline = (stripped.count("\"") % 2 == 1) or (stripped.count("(") != stripped.count(")"))
	
	if not current_chunk.is_empty():
		chunks.append(_trim_chunk(current_chunk.strip_edges(), context))
	
	# Final safety: ensure no chunk exceeds limit (fallback splitting)
	var final_chunks = PackedStringArray()
	for chunk in chunks:
		if chunk.length() > CHUNK_MAX_CHARS:
			final_chunks.append_array(_split_by_fixed_size(chunk))
		else:
			final_chunks.append(chunk)
	return final_chunks


# Fallback: split a text into fixed‑size overlapping chunks.
func _split_by_fixed_size(text: String, size: int = 800, overlap: int = 200) -> PackedStringArray:
	var chunks = PackedStringArray()
	var pos = 0
	while pos < text.length():
		var chunk = text.substr(pos, size).strip_edges()
		if not chunk.is_empty():
			chunks.append(chunk + "\n[... continued]")
		pos += (size - overlap)
	return chunks


# Trims a chunk if it still exceeds the hard limit and adds a context comment.
func _trim_chunk(chunk: String, context: String = "") -> String:
	if chunk.length() > CHUNK_MAX_CHARS:
		var trimmed = chunk.substr(0, CHUNK_MAX_CHARS - 50)
		return "# [TRUNCATED from %s]\n%s\n# [... truncated due to size limit]" % [context, trimmed]
	return chunk


# ==============================================================================
#   EMBEDDING FUNCTIONS
# ==============================================================================
func _request_chunk_embeddings(chunks: PackedStringArray):
	send_button.disabled = true
	"""Send all chunks to the embedding model in a single batch."""
	if chunks.size() == 0:
		message_display.append_text("[color=yellow]No chunks to embed.[/color]\n")
		return

	message_display.append_text("[i]Indexing %d chunks...[/i]\n" % chunks.size())
	_embedding_in_progress = true

	# Convert PackedStringArray to regular Array for JSON serialization
	var input_arr: Array = []
	for chunk in chunks:
		input_arr.append(chunk)
		
	var body_dict = {
		"model": EMBED_MODEL,
		"input": input_arr
	}
	var body = JSON.stringify(body_dict)
	var headers = ["Content-Type: application/json"]
	embed_request.request(EMBED_URL, headers, HTTPClient.METHOD_POST, body)


func _request_query_embedding(text: String):
	"""Embed a single user query."""
	var body_dict = {
		"model": EMBED_MODEL,
		"input": [text]
	}
	var body = JSON.stringify(body_dict)
	var headers = ["Content-Type: application/json"]
	embed_request.request(EMBED_URL, headers, HTTPClient.METHOD_POST, body)


func _on_embed_request_completed(result, response_code, headers, body):
	"""Handle responses from the embedding server."""
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		message_display.append_text("[color=red]Embedding request failed (%d)[/color]\n" % response_code)
		if _embedding_in_progress:
			# Batch indexing failure
			_embedding_in_progress = false
			send_button.disabled = false
		else:
			# Query embedding failure
			_abort_rag_chat()
		return

	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK:
		message_display.append_text("[color=red]Failed to parse embedding response[/color]\n")
		if _embedding_in_progress:
			_embedding_in_progress = false
			send_button.disabled = false
		else:
			_abort_rag_chat()
		return

	var data = json.get_data()
	if not data.has("embeddings"):
		message_display.append_text("[color=red]Unexpected embedding response format[/color]\n")
		if _embedding_in_progress:
			_embedding_in_progress = false
			send_button.disabled = false
		else:
			_abort_rag_chat()
		return

	var embeddings_list = data["embeddings"]   # Array of Arrays

	if not _embedding_in_progress:
		# This response was for a query embedding → proceed to retrieval
		if embeddings_list.size() > 0:
			var query_emb = embeddings_list[0]  # list of floats
			_retrieve_and_send_chat(query_emb)
		return

	# ---------- Batch indexing response ----------
	if embeddings_list.size() != _chunks.size():
		message_display.append_text("[color=yellow]Embedding count mismatch – indexing may be incomplete.[/color]\n")

	# Store all chunk embeddings as PackedFloat32Array for fast similarity calculations
	_chunk_embeddings.clear()
	for emb in embeddings_list:
		var packed = PackedFloat32Array(emb)   # ensure it's packed for arithmetic
		_chunk_embeddings.append(packed)

	_indexed = true
	_embedding_in_progress = false
	send_button.disabled = false
	message_display.append_text("[color=green]Indexing complete! %d chunks stored.[/color]\n" % _chunks.size())


# ==============================================================================
#   RETRIEVAL (COSINE SIMILARITY)
# ==============================================================================
func _cosine_similarity(a: PackedFloat32Array, b: PackedFloat32Array) -> float:
	if a.size() != b.size():
		return 0.0

	var dot := 0.0
	var norm_a := 0.0
	var norm_b := 0.0
	for i in range(a.size()):
		dot += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (sqrt(norm_a) * sqrt(norm_b))


func _retrieve_top_chunks(query_emb: PackedFloat32Array) -> Array[String]:
	"""Return top RETRIEVAL_TOP_K chunk texts using Maximum Marginal Relevance."""
	if _chunk_embeddings.is_empty() or _chunks.is_empty():
		return []
	
	var num_candidates: int = _chunk_embeddings.size()
	var top_k: int = min(RETRIEVAL_TOP_K, num_candidates)
	
	# Pre‑compute query similarity for all chunks (once)
	var query_sims: Array[float] = []
	for i in range(num_candidates):
		query_sims.append(_cosine_similarity(query_emb, _chunk_embeddings[i]))
	
	var selected: Array[int] = []
	var remaining = range(num_candidates)
	
	while selected.size() < top_k and remaining.size() > 0:
		var best_score: float = -1.0
		var best_idx: int = -1
		
		for i in remaining:
			# Relevance part
			var rel: float = query_sims[i]
			# Diversity part: max similarity to already selected chunks
			var max_sim_selected: float = 0.0
			for j in selected:
				var sim_ij: float = _cosine_similarity(_chunk_embeddings[i], _chunk_embeddings[j])
				if sim_ij > max_sim_selected:
					max_sim_selected = sim_ij
			# MMR score
			var mmr: float = MMR_LAMBDA * rel - (1.0 - MMR_LAMBDA) * max_sim_selected
			if mmr > best_score:
				best_score = mmr
				best_idx = i
		
		if best_idx != -1:
			selected.append(best_idx)
			remaining.erase(best_idx)
		
	# Convert selected indices to chunk texts
	var result: Array[String] = []
	for idx in selected:
		result.append(_chunks[idx])
	return result


func _retrieve_and_send_chat(query_emb: PackedFloat32Array):
	"""After query embedding is ready, retrieve chunks and build the chat request."""
	var relevant = _retrieve_top_chunks(query_emb)
	if relevant.is_empty():
		message_display.append_text("[color=yellow]Retrieval returned no chunks – using fallback.[/color]\n")

	# Build the conversation messages with RAG context
	var messages = _build_messages_with_rag(relevant)
	# Release the pending user text
	var user_text = _pending_user_text
	_pending_user_text = ""
	# Send the request
	_send_request(messages, false, true)


# ==============================================================================
#   BUILDING MESSAGES WITH RAG CONTEXT
# ==============================================================================
func _build_system_message_with_rag(chunks: Array[String]) -> String:
	"""Build a system prompt that includes only the relevant chunks."""
	var base = """You are Nindra, a Godot game engine expert assistant. Recognize and apply 
	design patterns (State, Observer, etc.). Be observant: scan provided code for 
	hidden issues (performance, fragile paths, memory leaks). 
	Proactively propose solutions, comparing alternatives when possible. 
	Always write complete, heavily commented code. Briefly state observations, 
	then implement the best solution. Use relevant project snippets; ask if insufficient.\n\n"""

	if chunks.size() > 0:
		base += "--- Relevant Snippets ---\n"
		for i in chunks.size():
			base += "[Snippet %d]\n%s\n\n" % [i+1, chunks[i]]
		base += "--- End of Snippets ---\n"
	else:
		base += "(No project context available.)\n"
	return base


func _build_messages_with_rag(chunks: Array[String]) -> Array:
	var messages := []
	var sys = _build_system_message_with_rag(chunks)
	if not sys.is_empty():
		messages.append({"role": "system", "content": sys})
	messages.append_array(conversation.duplicate(true))
	print("SYSTEM PROMPT:\n", sys)
	return messages


# ==============================================================================
#   BUTTON HANDLERS
# ==============================================================================

func _on_clear_pressed():
	conversation.clear()
	_conversation_display_text = ""
	message_display.text = ""
	_last_warning = ""
	message_display.append_text("[i]Conversation cleared.[/i]\n")


func _on_suggest_next_pressed():
	_send_code_task("Continue the following Godot script exactly where the cursor would be. Only output the code that should come next, without explanation.")


func _on_improve_pressed():
	_send_code_task("Review the following Godot script and suggest improvements, refactorings, or bugs. Explain your reasoning.")


var _full_context_embedded := false   # TODO: will be used when embedding is fully implemented
func _on_embed_pressed():
	"""Load export file, chunk it, and start indexing."""
	if _embedding_in_progress:
		message_display.append_text("[color=yellow]Indexing already in progress...[/color]\n")
		return

	var content = _load_context()
	if content == "":
		return

	_chunks = _chunk_code(content)
	
	if _chunks.size() == 0:
		message_display.append_text("[color=yellow]No chunks extracted from file.[/color]\n")
		return

	# Start the embedding process
	_request_chunk_embeddings(_chunks)


func _on_send_pressed():
	if _embedding_in_progress:
		message_display.append_text("[color=yellow]Indexing is still in progress. Please wait...[/color]\n")
		return
	var user_text = prompt_input.text.strip_edges()
	if user_text == "":
		return

	conversation.append({"role": "user", "content": user_text})
	_display_message("user", user_text)
	prompt_input.text = ""
	send_button.disabled = true
	message_display.append_text("[i]Nindra is thinking...[/i]\n")

	if _indexed and _chunks.size() > 0:
		# RAG pipeline: embed query first
		_pending_user_text = user_text
		_request_query_embedding(user_text)
	else:
		# Fallback: use old full‑file system message
		var messages = _build_messages_with_context()
		_send_request(messages, false, true)

# ------------------------------------------------------------------------------
# Limits the conversation array to the last MAX_HISTORY_PAIRS exchanges
# ------------------------------------------------------------------------------
func _trim_conversation() -> void:
	# Each turn is one user + one assistant → 2 entries
	var max_messages := MAX_HISTORY_PAIRS * 2
	while conversation.size() > max_messages:
		# Remove the oldest message (first element)
		conversation.pop_front()


# ==============================================================================
#   CORE REQUEST LOGIC
# ==============================================================================

# Builds the system message that uses CONTEXT_FILE_PATH
func _build_system_message_with_full_context() -> String:
	var context_text = _load_context()
	if context_text != "":
		return """You are Nindra, a Godot game engine expert assistant. Recognize and apply 
	design patterns (State, Observer, etc.). Be observant: scan provided code for 
	hidden issues (performance, fragile paths, memory leaks). 
	Proactively propose solutions, comparing alternatives when possible. 
	Always write complete, heavily commented code. Briefly state observations, 
	then implement the best solution. Use relevant project snippets; ask if insufficient.\n\n""" + context_text
	else:
		return """You are Nindra. a Godot game engine expert assistant. Recognize and apply 
	design patterns (State, Observer, etc.). Be observant: scan provided code for 
	hidden issues (performance, fragile paths, memory leaks). 
	Proactively propose solutions, comparing alternatives when possible. 
	Always write complete, heavily commented code. Briefly state observations, 
	then implement the best solution. Use relevant project snippets; ask if insufficient. (No project context loaded yet.)"""


# Normal chat handler builds messages with full context + conversation history
func _build_messages_with_context() -> Array:
	var messages := []
	var sys_text = _build_system_message_with_full_context()
	if not sys_text.is_empty():
		messages.append({"role": "system", "content": sys_text})
	messages.append_array(conversation.duplicate(true))
	return messages


#--------------------helper to abort the RAG chat cleanly-----------------------

func _abort_rag_chat():
	"""Clean up after a failed query embedding – remove the last user message and release the UI."""
	# Pop the user message we just added (it‘s the last one, and it’s a user role)
	if conversation.size() > 0 and conversation[-1]["role"] == "user":
		conversation.pop_back()
	_pending_user_text = ""
	send_button.disabled = false
	message_display.append_text("[color=red]Failed to process your request. Please try again.[/color]\n")


# ==============================================================================
#   CONTEXT FILE READER
# ==============================================================================

func _load_context() -> String:
	if not FileAccess.file_exists(CONTEXT_FILE_PATH):
		_display_system_warning("Project context file not found: " + CONTEXT_FILE_PATH)
		return ""
	
	var file = FileAccess.open(CONTEXT_FILE_PATH, FileAccess.READ)
	if file == null:
		_display_system_warning("Failed to open context file.")
		return ""
	
	var content = file.get_as_text()
	const MAX_SIZE = 1024 * 800 # 800 kb
	if content.length() > MAX_SIZE:
		_display_system_warning("Context file is very large; only first %d bytes sent." % MAX_SIZE)
		content = content.substr(0, MAX_SIZE) + "\n[... truncated]"
	return content


# ==============================================================================
#   HTTP REQUEST & RESPONSE HANDLER
# ==============================================================================

func _on_http_request_request_completed(result, response_code, headers, body):
	send_button.disabled = false
	
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		message_display.append_text("[color=red]Error: %s (code %d)[/color]\n" % [result, response_code])
		return
	
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		message_display.append_text("[color=red]Failed to parse JSON[/color]\n")
		return
	
	var data = json.get_data()
	if data.has("message") and data["message"].has("content"):
		var ai_text = data["message"]["content"].strip_edges()
		# Only add to conversation if it's a normal chat reply (not a code task)
		# We determine this by checking if the request came from a code task (no easy flag here).
		# For simplicity, we won't add code task answers to history – they are one‑offs.
		# We'll add a flag to _send_request to skip history insertion.
		# So we need to modify _send_request to accept a flag.
		# Let's implement that now.
		if not _skip_history_next:
			conversation.append({"role": "assistant", "content": ai_text})
			_trim_conversation()  
		_display_message("assistant", ai_text)
	else:
		message_display.append_text("[color=red]Unexpected response format[/color]\n")


# Modified _send_request to take a flag
func _send_request(messages: Array, skip_history := false, use_stream := true):
	_skip_history_next = skip_history
	var body = {
		"model": MODEL,
		"messages": messages,
		"stream": use_stream
	}
	var json_body = JSON.stringify(body)
	
	if use_stream:
		_start_streaming_request(json_body)
	else:
		var headers = ["Content-Type: application/json"]
		var error = http_request.request(OLLAMA_CHAT_URL, headers, HTTPClient.METHOD_POST, json_body)
		if error != OK:
			message_display.append_text("[color=red]HTTP request failed: %s[/color]\n" % error)
			send_button.disabled = false


var _streaming_request_path: String
var _streaming_request_headers: PackedStringArray
var _streaming_request_body: String
var _streaming_state: int = 0   # 0:idle, 1:connecting, 2:connected, 3:requesting, 4:reading

func _start_streaming_request(json_body: String):
	_streaming_client = HTTPClient.new()
	# Connect to ollama
	var err := _streaming_client.connect_to_host("127.0.0.1", 11434) # no http:// prefix!
	if err != OK:
		message_display.append_text("[color=red]Failed to connect: %s[/color]\n" % err)
		send_button.disabled = false
		_streaming_client = null
		return
	
	# Store the request for later
	_streaming_request_path = "/api/chat"
	_streaming_request_headers = [
		"Content-Type: application/json",
		"Accept: application/json"
	]
	_streaming_request_body = json_body
	
	# Set up display: show the AI prefix and start empty
	_streaming_response_text = ""
	_streaming_display_prefix = "🤖 Nindra:\n"
	_is_streaming = true
	_streaming_state = 1   # connecting
	_stream_done = false
	#print("[DEBUG] Streaming request started, process should poll now.")


func _send_code_task(task_description: String):
	var editor = EditorInterface.get_script_editor()
	var current_script = editor.get_current_script()
	if current_script == null:
		_display_system_warning("No script is currently open in the editor.")
		return
	
	var code_text = current_script.source_code
	if code_text.strip_edges() == "":
		_display_system_warning("The current script is empty.")
		return
	
	var messages := []
	var sys_msg = _build_system_message_with_full_context()
	if not sys_msg.is_empty():
		messages.append({"role": "system", "content": sys_msg})
	
	var user_msg = "%s\n\nHere is the script:\n```gdscript\n%s\n```" % [task_description, code_text]
	messages.append({"role": "user", "content": user_msg})
	
	message_display.append_text("[b]⚙ Task:[/b] %s\n" % task_description)
	message_display.append_text("[i]Nindra is thinking...[/i]\n")
	send_button.disabled = true
	
	_send_request(messages, true, false)   # skip history and no streaming for code tasks


# Normal chat: _on_send_pressed already calls _send_request(messages) with default skip_history = false
# So we must fix _on_send_pressed to call _send_request(messages, false) – the default so okay.

# Also fix _display_message for the system role (we added ⚙ but not used). Keep as before.

# ==============================================================================
#   DISPLAY & FORMATTING
# ==============================================================================

func _display_message(role: String, content: String):
	var formatted := _markdown_to_bbcode(content)
	var prefix := "🤖" if role == "assistant" else ("🧑" if role == "user" else "⚙")
	var line := "[b]%s %s:[/b]\n%s\n\n" % [prefix, role.capitalize(), formatted]
	
	# Append to our persistent base string
	_conversation_display_text += line
	
	# Show the full text on screen (will be overwritten if streaming is active)
	message_display.text = _conversation_display_text
	message_display.scroll_to_line(message_display.get_line_count() - 1)


func _display_system_warning(text: String):
	if text != _last_warning:
		_last_warning = text
		message_display.append_text("[color=yellow][i]⚠ %s[/i][/color]\n" % text)


func _markdown_to_bbcode(md: String) -> String:
	var regex = RegEx.new()
	regex.compile("```(?:.*?)\n(.*?)```")
	md = regex.sub(md, "[code]$1[/code]", true)
	
	regex.compile("`(.*?)`")
	md = regex.sub(md, "[code]$1[/code]", true)
	
	regex.compile("\\*\\*(.+?)\\*\\*")
	md = regex.sub(md, "[b]$1[/b]", true)
	
	regex.compile("(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)")
	md = regex.sub(md, "[i]$1[/i]", true)
	regex.compile("(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
	md = regex.sub(md, "[i]$1[/i]", true)
	
	regex.compile("(^|\n)\\s*(-|\\*)\\s+(.+)")
	md = regex.sub(md, "$1    • $3", true)
	
	return md


#-------------------------------------------------------------------------------
# AVATAR POP_UP LOGIC
#-------------------------------------------------------------------------------

func _on_avatar_gui_input(event: InputEvent):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		_show_avatar_popup()


func _show_avatar_popup():
	# Guard against a missing texture
	if avatar_texture.texture == null:
		return
	
	var popup = get_node_or_null("AvatarPopup")
	if popup == null:
		popup = Window.new()
		popup.name = "AvatarPopup"
		popup.title = ""
		popup.unresizable = true
		popup.popup_window = true
		popup.transparent = false
		popup.borderless = true
		var screen = DisplayServer.window_get_size()
		popup.size = screen * 0.7
		
		# Full‑screen dark background that also handles clicks
		var bg = ColorRect.new()
		bg.color = Color(0, 0, 0, 0.85)
		bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		bg.mouse_filter = Control.MOUSE_FILTER_STOP     # catches clicks
		popup.add_child(bg)
		
		# Big image
		var big_tex = TextureRect.new()
		big_tex.texture = avatar_texture.texture
		big_tex.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		big_tex.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		big_tex.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		big_tex.mouse_filter = Control.MOUSE_FILTER_STOP
		popup.add_child(big_tex)
		
		# Close when clicking the background or the image
		bg.gui_input.connect(_on_popup_clicked.bind(popup))
		big_tex.gui_input.connect(_on_popup_clicked.bind(popup))
		
		# Close with Escape
		popup.close_requested.connect(func(): popup.hide())
		
		add_child(popup)
	
	popup.popup_centered()


func _on_popup_clicked(event: InputEvent, popup: Window):
	if event is InputEventMouseButton and event.pressed:
		popup.hide()
