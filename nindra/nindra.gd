@tool
extends EditorPlugin

var chat_dock

func _enable_plugin() -> void:
	# Add autoloads here.
	pass


func _disable_plugin() -> void:
	# Remove autoloads here.
	pass


func _enter_tree() -> void:
	# Load and instance the chat dock
	chat_dock = preload("res://addons/nindra/chat_dock.tscn").instantiate()
	# Add it to the right side, next to the Inspector
	add_control_to_dock(DOCK_SLOT_RIGHT_UL, chat_dock)
	print("Nindra dock opened.")


func _exit_tree() -> void:
	# Clean-up of the plugin goes here.
	remove_control_from_docks(chat_dock)
	chat_dock.queue_free()
	print("Nindra dock closed.")
