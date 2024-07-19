package com.example.websocketchatapp.Model;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

@Service
public class ResponseService {
    private final SimpMessagingTemplate messagingTemplate;

    public ResponseService(SimpMessagingTemplate messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }

    public void sendGlobalNotification(String notification) {
        messagingTemplate.convertAndSend("/topic/global_notification", notification);
    }

    public void sendPrivateNotification(String userId, String notification) {
        messagingTemplate.convertAndSendToUser(userId, "/queue/private_notification", notification);
    }
}
