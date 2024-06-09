//
//  LoginViewController.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 09/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation
import UIKit

class LoginViewController: UIViewController {
    
    // Outlets for the text fields and the button
    @IBOutlet weak var usernameTextField: UITextField!
    @IBOutlet weak var passwordTextField: UITextField!
    @IBOutlet weak var loginButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // Action for the login button
    @IBAction func loginButtonTapped(_sender: UIButton) {
        guard let username = usernameTextField.text, !username.isEmpty,
              let password = passwordTextField.text, !password.isEmpty else {
            showAlert(message: "Please enter both username and password.")
            return
        }
        login(username: username, password: password)
    }
    
    // Function to show an alert
    func showAlert(message: String) {
        let alertController = UIAlertController(title: "Login Error", message: message, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "OK", style: .default))
        present(alertController, animated: true)
    }
    
    // Function to perform the login
     func login(username: String, password: String) {
         let loginRequestBody = LoginRequest(username: username, password: password)
         MapClient.login(responseType: AuthenticationResponse.self, body: loginRequestBody) { response, error in
             if let error = error {
                 DispatchQueue.main.async {
                     self.showAlert(message: "Login failed: \(error.localizedDescription)")
                 }
                 return
             }
             
             guard let response = response else {
                 DispatchQueue.main.async {
                     self.showAlert(message: "Login failed: No response from server.")
                 }
                 return
             }
             
             // Handle successful login here
             DispatchQueue.main.async {
                 self.performSegue(withIdentifier: "loginSuccess", sender: self)
             }
         }
     }
 }

struct LoginRequest: Codable {
    let username: String
    let password: String
}

