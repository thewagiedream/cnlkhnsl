package {{ cookiecutter.org_name_2 }}.{{ cookiecutter.package_name }}

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

class MyForegroundService : Service() {

    companion object {
        private const val CHANNEL_ID = "comfyui_service"
        private const val NOTIFICATION_ID = 1
        private const val TAG = "MyForegroundService"
    }

    override fun onCreate() {
        super.onCreate()

        Log.d(TAG, "onCreate called")

        createNotificationChannel()

        val notification: Notification =
            NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("ComfyUI")
                .setContentText("Running in background")
                .setSmallIcon(R.mipmap.ic_launcher)
                .setOngoing(true)
                .setForegroundServiceBehavior(
                    NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE
                )
                .build()

        startForeground(NOTIFICATION_ID, notification)

        Log.d(TAG, "Foreground service started")
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int
    ): Int {
        Log.d(TAG, "onStartCommand called")
        return START_STICKY
    }

    override fun onDestroy() {
        Log.d(TAG, "onDestroy called")
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {

            val channel = NotificationChannel(
                CHANNEL_ID,
                "ComfyUI Background Tasks",
                NotificationManager.IMPORTANCE_DEFAULT
            )

            channel.description = "Keeps ComfyUI running in the background"

            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)

            Log.d(TAG, "Notification channel created")
        }
    }
}