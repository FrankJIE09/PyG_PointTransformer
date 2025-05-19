import numpy as np
import os
from datetime import datetime
import time
import open3d as o3d  # Import Open3D for visualization
# Import pynput components
from pynput import keyboard
import threading  # Needed to run the pynput listener in a separate thread

# Import the necessary components from your orbbec_camera.py file
# Assuming orbbec_camera.py is located inside a 'camera' subdirectory
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers, initialize_all_connected_cameras, \
        close_connected_cameras
    # No longer need ESC_KEY if handling 'esc' via pynput Key object
    # ESC_KEY = 27
except ImportError as e:
    print(f"Error importing from camera.orbbec_camera: {e}")
    print("Please ensure 'orbbec_camera.py' is inside a 'camera' subdirectory and accessible in your Python path.")
    exit()

# Directory to save point clouds
SAVE_DIR = "saved_pointclouds"

# --- Global variable to store pressed keys ---
# Using a list to store characters of pressed keys
pressed_keys = []
# --- Flag to signal termination ---
stop_program_flag = threading.Event()  # Use an Event to signal termination across threads


# --- pynput callback function for key press ---
def on_press(key):
    global pressed_keys
    # print(f"Key pressed: {key}") # Debugging line

    try:
        # Handle alphanumeric keys
        char = key.char
        if char is not None:
            pressed_keys.append(char.lower())  # Store character, convert to lower case for case-insensitivity
    except AttributeError:
        # Handle special keys (like esc, space, etc.)
        if key == keyboard.Key.esc:
            print("\nEscape key pressed.")
            stop_program_flag.set()  # Signal the main thread to stop
        # elif key == keyboard.Key.space:
        #     pressed_keys.append('space') # Example: Handle space key
        # Add other special keys if needed


# --- pynput callback function for key release (optional for this task) ---
def on_release(key):
    # print(f"Key released: {key}") # Debugging line
    # No action needed on release for this program's logic
    pass


# --- Function to start the pynput listener thread ---
def start_key_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()  # Start the listener thread
    # We don't listener.join() here because we want the main thread to continue
    print("Keyboard listener started.")
    return listener  # Return the listener object so we can stop it later


def main():
    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # Get available serial numbers
    print("Searching for Orbbec cameras...")
    available_sns = get_serial_numbers()

    if not available_sns:
        print("No Orbbec cameras found. Exiting.")
        return

    # Initialize all found cameras
    cameras = initialize_all_connected_cameras(available_sns)

    if not cameras:
        print("Failed to initialize any cameras. Exiting.")
        return

    # Start streams for all initialized cameras
    for camera in cameras:
        try:
            # Start with depth, color, alignment, and sync enabled
            camera.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        except Exception as e:
            print(f"Failed to start stream for camera {camera.get_serial_number()}: {e}")

    # Filter out cameras that failed to start
    cameras = [c for c in cameras if c.stream]

    if not cameras:
        print("No camera streams started successfully. Exiting.")
        return

    # Use the first successfully initialized camera for the main loop
    main_camera = cameras[0]

    # --- Start the pynput keyboard listener thread ---
    key_listener = start_key_listener()
    # --- No dummy cv2 window needed anymore ---

    print("\n------------------------------------------------------")
    print(f"Camera {main_camera.get_serial_number()} is streaming and visualizing.")
    print(f"Press 'r' to save 6D point cloud to '{SAVE_DIR}' directory.")
    print("Press 'q' or 'esc' key anywhere to quit.")
    print("------------------------------------------------------")

    # Initialize Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Orbbec Camera {main_camera.get_serial_number()} Point Cloud")

    # Create an empty point cloud object and add it to the visualizer
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Flag to track if the point cloud has been added to the visualizer for the first time
    geometry_added_once = False

    try:
        # --- Main loop runs until the stop signal is set or window is closed ---
        while not stop_program_flag.is_set() and vis.poll_events():  # Also check if the Open3D window is still open
            # Get the latest 6D point cloud data (XYZRGB)
            points = main_camera.get_point_cloud(colored=True)

            if points is not None:
                if points.ndim == 2 and points.shape[1] == 6:  # Check if it's an Nx6 array (XYZRGB)
                    # Update the point cloud in the visualizer
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ coordinates
                    # Colors are typically 0-255, Open3D expects 0-1
                    colors = points[:, 3:6] / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    vis.update_geometry(pcd)
                    if not geometry_added_once:
                        vis.reset_view_point(True)  # Auto zoom on the first valid point cloud
                        geometry_added_once = True

                    vis.update_renderer()  # Update the rendering

                elif points.ndim == 2 and points.shape[1] == 3:  # Handle XYZ case if color fails
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.tile([0.7, 0.7, 0.7], (points.shape[0], 1)))  # Default gray
                    vis.update_geometry(pcd)
                    if not geometry_added_once:
                        vis.reset_view_point(True)  # Auto zoom on the first valid point cloud
                        geometry_added_once = True
                    vis.update_renderer()
                # If points is not None but has unexpected shape, poll/update visualizer anyway
                else:
                    vis.update_renderer()
            else:
                # If no point cloud is returned, still update the visualizer
                vis.update_renderer()

            # --- Process pressed keys from the global list ---
            global pressed_keys
            if pressed_keys:
                # Process a copy of the list and then clear the original
                keys_to_process = list(pressed_keys)
                pressed_keys.clear()  # Clear the list immediately

                for key_char in keys_to_process:
                    if key_char == 'r':
                        print("Attempting to save point cloud...")
                        # Get the latest point cloud for saving
                        points_to_save = main_camera.get_point_cloud(colored=True)
                        if points_to_save is not None:
                            if points_to_save.ndim == 2 and points_to_save.shape[1] == 6:  # Ensure it's 6D data
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Milliseconds
                                filename = os.path.join(SAVE_DIR,
                                                        f"pointcloud_{main_camera.get_serial_number()}_{timestamp}.txt")
                                try:
                                    # Save as text file
                                    # header = "X Y Z R G B"
                                    np.savetxt(filename, points_to_save, fmt='%.6f %.6f %.6f %d %d %d', delimiter=' ',
                                                comments='')
                                    print(f"Successfully saved {points_to_save.shape[0]} points to {filename}")
                                except Exception as e_save:
                                    print(f"Error saving point cloud: {e_save}")
                            else:
                                print(
                                    f"Warning: Point cloud for saving had unexpected shape {points_to_save.shape}, expected Nx6 (XYZRGB). Not saving.")
                        else:
                            print("Failed to get point cloud for saving.")

                    elif key_char == 'q':
                        print("Quit key 'q' pressed. Exiting.")
                        stop_program_flag.set()  # Signal the main thread to stop
                        break  # Break from key processing loop

            # Add a small sleep to prevent the loop from running too fast
            # This is important when using a non-blocking check like stop_program_flag.is_set()
            time.sleep(0.005)  # Sleep for 5 milliseconds


    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting loop.")
    except Exception as e_main:
        print(f"An unexpected error occurred: {e_main}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup: Stop the pipeline, close visualizer and keyboard listener
        print("\nCleaning up...")
        # Signal the stop_program_flag in case we exited the loop due to an error
        stop_program_flag.set()
        # Stop the keyboard listener thread
        if 'key_listener' in locals() and key_listener and key_listener.is_alive():
            key_listener.stop()
            # key_listener.join() # Optional: wait for the listener thread to finish

        # Close the Open3D visualizer window
        if 'vis' in locals() and vis:
            vis.destroy_window()

        # Close cameras
        close_connected_cameras(cameras)

        print("Program finished.")


if __name__ == "__main__":
    main()
