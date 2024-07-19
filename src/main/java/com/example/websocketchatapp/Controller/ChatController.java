package com.example.websocketchatapp.Controller;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.messaging.simp.SimpMessageSendingOperations;
import org.springframework.messaging.simp.annotation.SendToUser;
import org.springframework.stereotype.Controller;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessageHeaderAccessor;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Controller;
import com.example.websocketchatapp.Model.Chatmessage;

import javax.management.Notification;

@Controller
public class ChatController {
    private final SimpMessageSendingOperations messagingTemplate;

    public ChatController(SimpMessageSendingOperations messagingTemplate) {
        this.messagingTemplate = messagingTemplate;
    }
    @MessageMapping("/chat.send")
    @SendTo("/topic/public")
    public Chatmessage sendMessage(@Payload Chatmessage message) {
        // Logique de traitement du message ici
        return message;
    }

    @MessageMapping("/notification.send")
    @SendToUser("/queue/notifications")
    public Notification sendNotification(@Payload Notification notification) {
        // Logique de traitement de la notification ici
        messagingTemplate.convertAndSendToUser(notification.getMessage(), "/queue/notifications", notification);
        return notification;
    }
}
